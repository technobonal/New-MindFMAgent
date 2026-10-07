"""
agentes/agente_redactor_pedagogico.py

Nodo LangGraph del Redactor Pedagógico. Adapta el core del equipo
(redactar_pedagogicamente) al grafo de NuevaMente:

  state -> SolicitudAdaptacion -> core (async, generador Groq inyectado) -> state

Decisiones:
  - Abstención (EVIDENCIA_INSUFICIENTE, formato no disponible, parámetros
    faltantes o irreconocibles) y fallo del LLM terminan con status="error" y
    el motivo en `error`.
  - Salida inválida (ErrorSalidaInvalida) cuenta como un intento y reentra al
    loop de reintentos con el error como feedback. Al agotar los intentos
    termina con status="error".
  - NFR-SEC-02: lo que va a `error` (y de ahí a pantalla) es SIEMPRE un mensaje
    del catálogo (mensaje_usuario); el detalle técnico va al log.
  - El feedback del Revisor (o el error de validación) llega por
    state["feedback_redactor"] y se consume en el siguiente intento.
  - Los parámetros de personalización (perfil, formato, nicho, nivel) se
    normalizan con los enums del contrato (contracts/enums.py), que ya aceptan
    forma larga, tildes y mayúsculas. Este módulo NO mantiene tablas de alias.

Cliente HTTP:
  - GeneradorGroq abre un httpx.AsyncClient propio por llamada (async with), de
    modo que nace y muere en el event loop activo. Evita "Event loop is closed".

Variable de entorno opcional:
  - REDACTOR_STRUCTURED_METHOD = function_calling (default) | json_schema
    Método de with_structured_output. Si el 400 de Groq se debe al schema
    estricto, function_calling lo tolera; json_schema es el comportamiento
    anterior.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import PurePosixPath

from pydantic import BaseModel, ValidationError

from agent_state import AgentState
from agentes.especificacion_pedagogica import obtener_especificacion
from llm_client import crear_http_async_client, get_llm
from seguridad.rate_limiter import RateLimiter
from src.contracts import SolicitudAdaptacion
from src.contracts.enums import FormatoSalida, NichoSector, NivelDetalle, PerfilDestinatario
from src.errores import (
    MENSAJE_POR_DEFECTO,
    CodigoError,
    ErrorFormatoNoDisponible,
    ErrorLLM,
    ErrorSalidaInvalida,
)
from src.prompts.redactor_pedagogico import detectar_idioma

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

#: Perfil por defecto de producto cuando el Supervisor no fijó ninguno.
PERFIL_POR_DEFECTO = PerfilDestinatario.DESARROLLADOR.value

_METODOS_VALIDOS = {"function_calling", "json_schema"}
METODO_STRUCTURED_OUTPUT = os.getenv("REDACTOR_STRUCTURED_METHOD", "function_calling").strip()
if METODO_STRUCTURED_OUTPUT not in _METODOS_VALIDOS:
    log.warning(
        "REDACTOR_STRUCTURED_METHOD=%r no es válido (%s); se usa 'function_calling'.",
        METODO_STRUCTURED_OUTPUT,
        ", ".join(sorted(_METODOS_VALIDOS)),
    )
    METODO_STRUCTURED_OUTPUT = "function_calling"


# --------------------------------------------------------------------------
# Normalización de parámetros con los enums del contrato
# --------------------------------------------------------------------------

class ParametroNoReconocido(ValueError):
    """Un parámetro obligatorio vino con un valor que el contrato no reconoce."""

    def __init__(self, campo: str, valor: str, aceptados: list[str]):
        self.campo = campo
        self.valor = valor
        super().__init__(
            f"{campo}={valor!r} no reconocido. Valores aceptados: {', '.join(aceptados)}."
        )


def _canonico(
    enum_cls,
    valor,
    *,
    campo: str,
    default: str | None = None,
    estricto: bool = True,
) -> str | None:
    """
    Devuelve el valor canónico del enum del contrato.

      - Vacío/None            -> `default`.
      - Reconocido (con alias, tildes o mayúsculas) -> su valor canónico.
      - Irreconocible:
          estricto=True  -> ParametroNoReconocido (no se reemplaza en silencio).
          estricto=False -> `default`, dejando un warning en el log.
    """
    if valor is None or not str(valor).strip():
        return default
    texto = str(valor).strip()
    try:
        return enum_cls(texto).value
    except ValueError:
        if estricto:
            raise ParametroNoReconocido(campo, texto, enum_cls.valores_aceptados()) from None
        log.warning(
            "Parámetro opcional %s=%r no reconocido; se usa el default del contrato.",
            campo,
            texto,
        )
        return default


# --------------------------------------------------------------------------
# Generador estructurado (implementa GeneradorEstructurado del core)
# --------------------------------------------------------------------------

def _loguear_error_llm(exc: Exception) -> None:
    """Deja en el log el status y el BODY de la respuesta de Groq (clave para diagnosticar el 400)."""
    cuerpo = getattr(exc, "body", None)
    if cuerpo is None:
        cuerpo = getattr(getattr(exc, "response", None), "text", None)
    log.error(
        "GeneradorGroq: excepción en ainvoke: %s | tipo=%s | status=%s | body=%s",
        str(exc),
        type(exc).__name__,
        getattr(exc, "status_code", None),
        cuerpo,
        exc_info=True,
    )

_CAMPOS_RAIZ_RESPUESTA = {"estado", "contenido", "motivo_abstencion"}


def _recuperar_de_tool_use_failed(exc: Exception) -> dict | None:
    """
    Si Groq rechazó la llamada porque el modelo aplanó la estructura (campos de
    `contenido` en la raíz), re-anida lo generado en vez de descartarlo.
    Devuelve None si no hay nada recuperable.
    """
    cuerpo = getattr(exc, "body", None)
    if not isinstance(cuerpo, dict):
        return None
    fallida = cuerpo.get("failed_generation") or (cuerpo.get("error") or {}).get("failed_generation")
    if not isinstance(fallida, str):
        return None
    try:
        llamada = json.loads(fallida)
        args = llamada.get("arguments") if isinstance(llamada, dict) else None
        if isinstance(args, str):
            args = json.loads(args)
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    if "contenido" in args:
        return args
    raiz = {k: v for k, v in args.items() if k in _CAMPOS_RAIZ_RESPUESTA}
    contenido = {k: v for k, v in args.items() if k not in _CAMPOS_RAIZ_RESPUESTA}
    raiz["contenido"] = contenido or None
    return raiz


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

        # El cliente httpx nace y muere dentro del event loop activo: sin conexiones
        # heredadas de loops cerrados (causa de "Event loop is closed").
        async with crear_http_async_client() as http_client:
            llm = get_llm(
                mensajes,
                self._rate_limiter,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                http_async_client=http_client,
            )

            try:
                resultado = await llm.with_structured_output(
                    output_model,
                    include_raw=True,
                    method=METODO_STRUCTURED_OUTPUT,
                ).ainvoke(mensajes)
            except Exception as e:
                _loguear_error_llm(e)
                recuperado = _recuperar_de_tool_use_failed(e)
                if recuperado is not None:
                    log.warning("Redactor: salida recuperada de un tool_use_failed (estructura aplanada).")
                    return recuperado
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
        # Perfil ausente -> default de producto. Perfil irreconocible -> error
        # (PARAMETRO_INVALIDO): no se reemplaza en silencio por otro perfil.
        "perfil_destinatario": _canonico(
            PerfilDestinatario,
            state.get("perfil_destinatario"),
            campo="perfil_destinatario",
            default=PERFIL_POR_DEFECTO,
            estricto=True,
        ),
        # Sin default: un formato ausente o irreconocible falla la validación del
        # contrato (PARAMETRO_INVALIDO) en vez de generar un Tutorial no pedido.
        "formato_salida": _canonico(
            FormatoSalida,
            state.get("formato_salida"),
            campo="formato_salida",
            default="",
            estricto=True,
        ),
    }

    # nicho y nivel_detalle solo modulan: si vienen vacíos o irreconocibles,
    # rige el default del contrato (General / Estandar).
    nicho = _canonico(
        NichoSector,
        state.get("nicho_sector"),
        campo="nicho_sector",
        default=None,
        estricto=False,
    )
    nivel = _canonico(
        NivelDetalle,
        state.get("nivel_detalle"),
        campo="nivel_detalle",
        default=None,
        estricto=False,
    )
    if nicho:
        datos["nicho_sector"] = nicho
    if nivel:
        datos["nivel_detalle"] = nivel
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
        except ParametroNoReconocido as exc:
            return _error(MENSAJE_POR_DEFECTO[CodigoError.PARAMETRO_INVALIDO], str(exc))
        except ValidationError as exc:
            return _error(
                MENSAJE_POR_DEFECTO[CodigoError.PARAMETRO_INVALIDO], _resumir_validacion(exc)
            )

        chunks = [
            {"id": c["id"], "texto": c["texto"]}
            for c in (state.get("chunks_fuente_confirmados") or [])
        ]

        # Idioma de salida (misma heurística que usa el prompt).
        tema = state.get("tema_pedido_chat") or state.get("tema_consulta") or ""
        textos_chunks = [(c.get("texto") or "")[:400] for c in chunks]
        idioma_salida = detectar_idioma(tema, *textos_chunks[:3])

        try:
            resultado = await redactar_pedagogicamente(
                solicitud=solicitud,
                chunks=chunks,
                especificacion_pedagogica=obtener_especificacion(solicitud.perfil_destinatario),
                generador=generador,
                feedback_revisor=state.get("feedback_redactor"),
                idioma_salida=idioma_salida,
                tema_usuario=tema,
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
            **construir_salida_estado(solicitud, resultado.contenido, idioma_salida=idioma_salida),
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