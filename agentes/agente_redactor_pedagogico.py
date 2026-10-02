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
import re

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

TEMPERATURA_REDACTOR = 0.1
MAX_TOKENS_REDACTOR = 4096
MAX_CHARS_FEEDBACK = 1500
MENSAJE_EVIDENCIA_INSUFICIENTE = (
    "El documento no tiene evidencia suficiente para generar este contenido en este formato. "
    "Proba con otro formato (Resumen Ejecutivo, Flashcards o Quiz), otro tema u otro documento."
)

# ← ACÁ: mapas + _canon
PERFIL_MAP = {
    "principiante": "Principiante",
    "principiante/transición de carrera": "Principiante",
    "principiante/transicion de carrera": "Principiante",
    "desarrollador": "Desarrollador",
    "desarrollador junior/semi senior": "Desarrollador",
    "desarrollador junior": "Desarrollador",
    "lider tecnico": "Lider Tecnico",
    "líder técnico": "Lider Tecnico",
    "lider técnico": "Lider Tecnico",
    "líder técnico/arquitecto": "Lider Tecnico",
    "lider tecnico/arquitecto": "Lider Tecnico",
    "gestor ejecutivo": "Gestor Ejecutivo",
    "gestor/ejecutivo no técnico": "Gestor Ejecutivo",
    "gestor/ejecutivo no tecnico": "Gestor Ejecutivo",
}

FORMATO_MAP = {
    "flashcards": "Flashcards",
    "tutorial": "Tutorial",
    "guía práctica paso a paso": "Tutorial",
    "guia practica paso a paso": "Tutorial",
    "quiz": "Quiz",
    "quiz interactivo con justificaciones": "Quiz",
    "resumen ejecutivo": "Resumen Ejecutivo",
    "resumen ejecutivo (tl;dr)": "Resumen Ejecutivo",
    "guion de clase": "Guion de Clase",
    "guión de clase": "Guion de Clase",
    "guion de clase/video": "Guion de Clase",
    "guión de clase/video": "Guion de Clase",
}

NICHO_MAP = {
    "fintech": "Fintech",
    "salud": "Salud",
    "e-commerce": "E-commerce",
    "ecommerce": "E-commerce",
    "general": "General",
}

def _canon(valor: str | None, tabla: dict[str, str], default: str) -> str:
    if not valor or not str(valor).strip():
        return default
    key = str(valor).strip().lower()
    return tabla.get(key, default)


def _detectar_idioma(*textos: str | None) -> str:
    """Heurística liviana. Prioriza el tema del usuario si está presente."""
    
    muestra = " ".join(t for t in textos if t).strip().lower()
    if not muestra:
        return "es"
    # Español
    if re.search(r"[áéíóúñ¿¡]", muestra) or any(
        w in f" {muestra} " for w in (" qué ", " cómo ", " para ", " sobre ", " del ", " una ", " los ")
    ):
        return "es"
    # Portugués
    if re.search(r"[ãõç]", muestra) or "ção" in muestra or "ões" in muestra:
        return "pt"
    # Inglés
    if any(
        w in f" {muestra} " for w in (" the ", " and ", " of ", " for ", " with ", " what ", " how ")
    ):
        return "en"
    return "es"


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

        try:
            resultado = await llm.with_structured_output(
                output_model, include_raw=True
            ).ainvoke(mensajes)
        except Exception as e:
            log.error(
                "GeneradorGroq: excepción en ainvoke: %s | tipo=%s",
                str(e),
                type(e).__name__,
                exc_info=True,
            )
            raise  # sube al core y se convierte en ErrorLLM

        if resultado.get("parsed") is not None:
            return resultado["parsed"]

        # Structured output falló → logueamos el raw completo para diagnosticar
        parsing_error = resultado.get("parsing_error")
        raw_msg = resultado.get("raw")
        crudo = getattr(raw_msg, "content", None) or str(raw_msg) or ""

        log.warning(
            "Redactor: parseo estructurado falló.\n"
            "parsing_error: %s\n"
            "raw content (primeros 2000 chars):\n%s",
            parsing_error,
            crudo[:2000],
        )

        # Intentamos recuperar algo usable
        try:
            intento = json.loads(crudo) if isinstance(crudo, str) else crudo
            if isinstance(intento, dict):
                return intento
        except Exception:
            pass

        # Si no se pudo recuperar nada, lanzamos excepción clara
        raise ValueError(
            f"Structured output falló y no se pudo recuperar JSON válido. "
            f"parsing_error={parsing_error}"
        )


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
        "documento_contenido": "\n\n".join(
            (c.get("texto") or "") for c in chunks if isinstance(c, dict)
        ),
        # Defaults de producto si el Supervisor no los fijó / vinieron raros
        "perfil_destinatario": _canon(
            state.get("perfil_destinatario"), PERFIL_MAP, "Desarrollador"
        ),
        # Sin default: un formato ausente o irreconocible falla la validación del
        # contrato (PARAMETRO_INVALIDO) en vez de generar un Tutorial no pedido.
        "formato_salida": _canon(
            state.get("formato_salida"), FORMATO_MAP, ""
        ),
    }
    nicho = _canon(state.get("nicho_sector"), NICHO_MAP, "General")
    # nicho/nivel opcionales en el contrato: solo si el enum los acepta
    nicho_ok = _opcional_valido(NichoSector, nicho)
    nivel_ok = _opcional_valido(NivelDetalle, state.get("nivel_detalle"))
    if nicho_ok:
        datos["nicho_sector"] = nicho_ok
    if nivel_ok:
        datos["nivel_detalle"] = nivel_ok
    return SolicitudAdaptacion.model_validate(datos)


def construir_salida_estado(solicitud, contenido, idioma_salida: str = "es") -> dict:
    """ContenidoRedactor (core) -> campos del state. Lo reusa el Modificador."""
    adaptado = {
        "titulo": contenido.titulo,
        "introduccion_contextualizada": contenido.introduccion_contextualizada,
        "items": [item.model_dump(mode="json") for item in contenido.items],
    }
    # Campos nuevos del estándar ejecutivo (si el modelo los expone)
    if getattr(contenido, "mensaje_principal", None):
        adaptado["mensaje_principal"] = contenido.mensaje_principal
    if getattr(contenido, "recomendaciones_prioritarias", None):
        adaptado["recomendaciones_prioritarias"] = list(contenido.recomendaciones_prioritarias or [])

    return {
        "contenido_adaptado": adaptado,
        "metadatos": {
            "perfil_aplicado": solicitud.perfil_destinatario.value,
            "formato_generado": solicitud.formato_salida.value,
            "tiempo_estimado_estudio_minutos": contenido.tiempo_estimado_estudio_minutos,
            "conceptos_clave": contenido.conceptos_clave,
            "nicho_aplicado": solicitud.nicho_sector.value,
            "nivel_detalle_aplicado": solicitud.nivel_detalle.value,
            "prerrequisitos": contenido.prerrequisitos,
            "idioma": idioma_salida,
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

        # ─── NUEVO: detectar idioma de salida ───────────────────────────
        tema = state.get("tema_pedido_chat") or state.get("tema_consulta") or ""
        textos_chunks = [(c.get("texto") or "")[:400] for c in chunks]
        idioma_salida = _detectar_idioma(tema, *textos_chunks[:3])
        # ────────────────────────────────────────────────────────────────

        try:
            resultado = await redactar_pedagogicamente(
                solicitud=solicitud,
                chunks=chunks,
                especificacion_pedagogica=obtener_especificacion(solicitud.perfil_destinatario),
                generador=generador,
                feedback_revisor=state.get("feedback_redactor"),
                # ─── NUEVO: pasar idioma y tema al core ─────────────────
                idioma_salida=idioma_salida,
                tema_usuario=tema,
                # ────────────────────────────────────────────────────────
            )
        except ErrorFormatoNoDisponible as exc:
            return _error(exc.mensaje_usuario, exc.mensaje_tecnico)
        except ErrorSalidaInvalida as exc:
            if intentos >= max_intentos:
                return _error(exc.mensaje_usuario, f"intentos agotados ({intentos}): {exc.mensaje_tecnico}")
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
            # ─── NUEVO: idioma en metadatos ─────────────────────────────
            **construir_salida_estado(solicitud, resultado.contenido, idioma_salida=idioma_salida),
            # ────────────────────────────────────────────────────────────
            "feedback_redactor": None,
            "intentos_redactor": intentos,
        }

    return nodo_redactor


def enrutar_tras_redactor(state: AgentState) -> str:
    if state.get("status") == "error":
        return "abstencion"
    if not state.get("contenido_adaptado"):  # salida inválida, quedan intentos
        return "reintentar"
    return "revisar"