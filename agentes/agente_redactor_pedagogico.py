"""
agentes/agente_redactor_pedagogico.py

Nodo LangGraph del Redactor Pedagógico. Adapta el core del equipo
(redactar_pedagogicamente) al grafo de NuevaMente:

  state -> SolicitudAdaptacion -> core (async, generador Groq inyectado) -> state

Decisiones:
  - Abstención (EVIDENCIA_INSUFICIENTE, formato no disponible, parámetros
    faltantes) y fallo del LLM terminan con status="error" y el motivo en `error`.
  - Salida inválida (ErrorSalidaInvalida) cuenta como un intento y reentra al
    loop de reintentos con el error como feedback. Al agotar los intentos
    termina con status="error".
  - NFR-SEC-02: lo que va a `error` (y de ahí a pantalla) es SIEMPRE un mensaje
    del catálogo (mensaje_usuario); el detalle técnico va al log.
  - El feedback del Revisor (o el error de validación) llega por
    state["feedback_redactor"] y se consume en el siguiente intento.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import PurePosixPath

from pydantic import BaseModel, ValidationError

from agent_state import AgentState
from agentes.especificacion_pedagogica import obtener_especificacion
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter
from src.contracts import SolicitudAdaptacion
from src.contracts.enums import NichoSector, NivelDetalle
from src.errores import (
    MENSAJE_POR_DEFECTO,
    CodigoError,
    ErrorFormatoNoDisponible,
    ErrorLLM,
    ErrorSalidaInvalida,
)

# AJUSTAR esta línea a la ruta real del módulo que define redactar_pedagogicamente.
from src.agentes.agente_redactor_pedagogico import EstadoRedactor, redactar_pedagogicamente

log = logging.getLogger(__name__)

TEMPERATURA_REDACTOR = 0.1  # generación estructurada: conviene estable
# get_llm usa max_tokens=1024 por defecto: corta el JSON de un paquete completo.
# Ajustar según el TPM real de la cuenta de Groq.
MAX_TOKENS_REDACTOR = 4096
MAX_CHARS_FEEDBACK = 1500
MENSAJE_EVIDENCIA_INSUFICIENTE = (
    "No hay evidencia suficiente en el documento para generar este contenido. "
    "Proba con otro tema o con otro documento."
)


# --------------------------------------------------------------------------
# Generador estructurado (implementa GeneradorEstructurado del core)
# --------------------------------------------------------------------------

class GeneradorGroq:
    def __init__(
        self,
        rate_limiter: RateLimiter,
        temperature: float = TEMPERATURA_REDACTOR,
        max_tokens: int = MAX_TOKENS_REDACTOR,
    ):
        self._rate_limiter = rate_limiter
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, *, prompt: str, output_model: type[BaseModel]):
        mensajes = [{"role": "user", "content": prompt}]
        # get_llm puede bloquear (rate limiter): fuera del event loop.
        llm = await asyncio.to_thread(
            get_llm,
            mensajes,
            self._rate_limiter,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        resultado = await llm.with_structured_output(output_model, include_raw=True).ainvoke(mensajes)

        if resultado.get("parsed") is not None:
            return resultado["parsed"]

        # El LLM respondió pero el parseo falló (JSON truncado, enum inválido...).
        # El core envuelve CUALQUIER excepción del generador en ErrorLLM (terminal);
        # para que esto entre al loop de reintentos como ErrorSalidaInvalida, se
        # devuelve lo que el modelo produjo y el core lo valida con mensaje preciso.
        log.info("Redactor: parseo estructurado falló: %s", resultado.get("parsing_error"))
        crudo = getattr(resultado.get("raw"), "content", "") or ""
        try:
            intento = json.loads(crudo)
        except (TypeError, ValueError):
            return {}
        return intento if isinstance(intento, dict) else {}


# --------------------------------------------------------------------------
# state -> SolicitudAdaptacion
# --------------------------------------------------------------------------

def _titulo_documento(objeto_id: str | None) -> str:
    """'fuentes/manual_vcn.pdf' -> 'manual_vcn'."""
    nombre = PurePosixPath(objeto_id or "").stem
    return nombre or "Documento"


def _opcional_valido(enum_cls, valor):
    """nicho/nivel solo modulan: si el Supervisor devolvió algo irreconocible, se usa el default."""
    if not valor:
        return None
    try:
        enum_cls(valor)
    except ValueError:
        return None
    return valor


def _resumir_validacion(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'solicitud'}: {e['msg']}" for e in exc.errors()
    )


def _construir_solicitud(state: AgentState) -> SolicitudAdaptacion:
    chunks = state.get("chunks_fuente_confirmados") or []
    datos = {
        "documento_titulo": _titulo_documento(state.get("objeto_id_confirmado")),
        # El contrato exige el texto, pero en este flujo vive en Chroma: se usan
        # los chunks confirmados, que son la fuente que el Redactor realmente ve.
        "documento_contenido": "\n\n".join(c["texto"] for c in chunks),
        "perfil_destinatario": state.get("perfil_destinatario"),
        "formato_salida": state.get("formato_salida"),
    }
    nicho = _opcional_valido(NichoSector, state.get("nicho_sector"))
    nivel = _opcional_valido(NivelDetalle, state.get("nivel_detalle"))
    if nicho:
        datos["nicho_sector"] = nicho
    if nivel:
        datos["nivel_detalle"] = nivel
    return SolicitudAdaptacion.model_validate(datos)


def construir_salida_estado(solicitud, contenido) -> dict:
    """ContenidoRedactor (core) -> campos del state. Lo reusa el Modificador."""
    return {
        "contenido_adaptado": {
            "titulo": contenido.titulo,
            "introduccion_contextualizada": contenido.introduccion_contextualizada,
            "items": [item.model_dump(mode="json") for item in contenido.items],
        },
        "metadatos": {
            "perfil_aplicado": solicitud.perfil_destinatario.value,
            "formato_generado": solicitud.formato_salida.value,
            "tiempo_estimado_estudio_minutos": contenido.tiempo_estimado_estudio_minutos,
            "conceptos_clave": contenido.conceptos_clave,
            "nicho_aplicado": solicitud.nicho_sector.value,
            "nivel_detalle_aplicado": solicitud.nivel_detalle.value,
            "prerrequisitos": contenido.prerrequisitos,
        },
    }


# --------------------------------------------------------------------------
# Nodo
# --------------------------------------------------------------------------

def construir_nodo_redactor(rate_limiter: RateLimiter, max_intentos: int):
    generador = GeneradorGroq(rate_limiter)

    async def nodo_redactor(state: AgentState) -> dict:
        intentos = state.get("intentos_redactor", 0) + 1

        def _error(mensaje_usuario: str, tecnico: str = "") -> dict:
            if tecnico:
                log.warning("Redactor abortó: %s", tecnico)
            return {
                "status": "error",
                "error": mensaje_usuario,
                "contenido_adaptado": {},
                "intentos_redactor": intentos,
            }

        try:
            solicitud = _construir_solicitud(state)
        except ValidationError as exc:
            return _error(
                MENSAJE_POR_DEFECTO[CodigoError.PARAMETRO_INVALIDO], _resumir_validacion(exc)
            )

        chunks = [
            {"id": c["id"], "texto": c["texto"]}
            for c in (state.get("chunks_fuente_confirmados") or [])
        ]

        try:
            resultado = await redactar_pedagogicamente(
                solicitud=solicitud,
                chunks=chunks,
                especificacion_pedagogica=obtener_especificacion(solicitud.perfil_destinatario),
                generador=generador,
                feedback_revisor=state.get("feedback_redactor"),
            )
        except ErrorFormatoNoDisponible as exc:
            return _error(exc.mensaje_usuario, exc.mensaje_tecnico)
        except ErrorSalidaInvalida as exc:
            if intentos >= max_intentos:
                return _error(exc.mensaje_usuario, f"intentos agotados ({intentos}): {exc.mensaje_tecnico}")
            # Reentra al loop: el detalle técnico viaja como feedback al LLM
            # (nunca a pantalla) en el próximo intento.
            log.info("Redactor: salida inválida en intento %s: %s", intentos, exc.mensaje_tecnico)
            return {
                "contenido_adaptado": {},
                "feedback_redactor": exc.mensaje_tecnico[:MAX_CHARS_FEEDBACK],
                "intentos_redactor": intentos,
            }
        except ErrorLLM as exc:
            return _error(exc.mensaje_usuario, exc.mensaje_tecnico)

        if resultado.estado is EstadoRedactor.EVIDENCIA_INSUFICIENTE:
            return _error(MENSAJE_EVIDENCIA_INSUFICIENTE, resultado.motivo_abstencion or "")

        return {
            **construir_salida_estado(solicitud, resultado.contenido),
            "feedback_redactor": None,  # ya consumido
            "intentos_redactor": intentos,
        }

    return nodo_redactor


def enrutar_tras_redactor(state: AgentState) -> str:
    if state.get("status") == "error":
        return "abstencion"
    if not state.get("contenido_adaptado"):  # salida inválida, quedan intentos
        return "reintentar"
    return "revisar"