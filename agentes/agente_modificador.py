"""
agentes/agente_modificador.py

Agente Modificador: reemplaza al Redactor en la vuelta de "¿modificar algo?".
Reusa el core del Redactor del equipo (mismo schema, mismas validaciones de
anclaje): la instrucción del usuario y la versión actual viajan en el campo
`feedback_revisor` del core, y el resultado pasa por el MISMO Revisor.

Contrato de contadores (ver grafo.py): el Modificador incrementa
intentos_modificador en cada ejecución; vueltas_modificacion lo lleva el grafo.

Si el cambio NO se puede aplicar (evidencia insuficiente, salida inválida tras
1 reintento interno, fallo del LLM) el contenido actual NO se toca: se devuelve
`aviso_modificacion` (mensaje para el usuario, NFR-SEC-02) y el router decide:
  - 1ra ejecución de la instrucción -> "sin_cambio": vuelve a preguntar al usuario.
  - ejecuciones posteriores (ya había una versión rechazada por el Revisor)
    -> "guardar": se guarda lo último con advertencias.

Además, en la 1ra ejecución busca chunks NUEVOS relacionados con la instrucción
(umbral del Investigador), porque el usuario puede pedir algo que los chunks
originales no cubren ("agregá una flashcard sobre X").
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from agent_state import AgentState
from agentes.agente_investigador import UMBRAL_SIMILITUD
from agentes.agente_redactor_pedagogico import (
    MAX_CHARS_FEEDBACK,
    MENSAJE_POR_DEFECTO,
    CodigoError,
    ErrorFormatoNoDisponible,
    ErrorLLM,
    ErrorSalidaInvalida,
    EstadoRedactor,
    GeneradorGroq,
    _construir_solicitud,
    _resumir_validacion,
    construir_salida_estado,
    redactar_pedagogicamente,
)
from agentes.especificacion_pedagogica import obtener_especificacion
from seguridad.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

TOP_K_EXTRA = 3                      # chunks nuevos máximos por instrucción
MAX_REINTENTOS_SALIDA_INVALIDA = 1   # reintentos internos por salida inválida
MENSAJE_SIN_EVIDENCIA = (
    "No encontré en el documento respaldo suficiente para aplicar ese cambio. "
    "Probá con otra instrucción."
)

INDICACION = (
    "El usuario pidió MODIFICAR el material ya generado. Devolvé el paquete COMPLETO "
    "aplicando únicamente este cambio: conservá el resto salvo que la instrucción lo afecte, "
    "mantené el mismo formato y usá solo hechos de los chunks (cada item con anclaje a IDs "
    "de chunks existentes). Si la instrucción pide algo que los chunks no respaldan, abstenete."
)


def construir_nodo_modificador(rate_limiter: RateLimiter, buscar_chunks):
    """buscar_chunks: BuscarChunksEnAlcance (consulta, k, user_id, objeto_id) -> [(id, texto, sim)]."""
    generador = GeneradorGroq(rate_limiter)

    async def nodo_modificador(state: AgentState, config: RunnableConfig) -> dict:
        user_id = str(config["configurable"]["thread_id"])
        objeto_id = state.get("objeto_id_confirmado") or ""
        instruccion = state.get("instruccion_modificacion") or ""
        intentos_previos = state.get("intentos_modificador", 0)
        intentos = intentos_previos + 1

        def _sin_cambio(mensaje_usuario: str, tecnico: str = "") -> dict:
            if tecnico:
                log.warning("Modificador no aplicó el cambio: %s", tecnico)
            # NO se toca contenido_adaptado: sigue la última versión válida.
            return {
                "aviso_modificacion": mensaje_usuario,
                "feedback_redactor": None,
                "intentos_modificador": intentos,
            }

        # ---- chunks: los confirmados + los que la instrucción pueda necesitar
        chunks = list(state.get("chunks_fuente_confirmados") or [])
        chunks_nuevos = False
        if intentos_previos == 0:
            try:
                encontrados = await asyncio.to_thread(
                    buscar_chunks, instruccion, TOP_K_EXTRA, user_id, objeto_id
                )
                ids = {c["id"] for c in chunks}
                extra = [
                    {"id": i, "texto": t}
                    for i, t, sim in encontrados
                    if sim >= UMBRAL_SIMILITUD and i not in ids
                ]
                if extra:
                    chunks += extra
                    chunks_nuevos = True
            except Exception as exc:  # la búsqueda extra es opcional
                log.warning("Modificador: falló la búsqueda de chunks extra: %s", exc)

        try:
            solicitud = _construir_solicitud({**state, "chunks_fuente_confirmados": chunks})
        except ValidationError as exc:
            return _sin_cambio(
                MENSAJE_POR_DEFECTO[CodigoError.PARAMETRO_INVALIDO], _resumir_validacion(exc)
            )

        # Feedback del Revisor solo si es un reintento de ESTA instrucción
        # (en la 1ra ejecución puede haber feedback viejo de otra vuelta).
        feedback = {
            "tipo": "MODIFICACION_SOLICITADA_POR_EL_USUARIO",
            "instruccion_del_usuario": instruccion,
            "version_actual": state.get("contenido_adaptado") or {},
            "indicacion": INDICACION,
        }
        if intentos_previos > 0 and state.get("feedback_redactor"):
            feedback["correcciones_pendientes_del_revisor"] = state["feedback_redactor"]

        chunks_core = [{"id": c["id"], "texto": c["texto"]} for c in chunks]
        especificacion = obtener_especificacion(solicitud.perfil_destinatario)

        resultado = None
        for llamada in range(1 + MAX_REINTENTOS_SALIDA_INVALIDA):
            try:
                resultado = await redactar_pedagogicamente(
                    solicitud=solicitud,
                    chunks=chunks_core,
                    especificacion_pedagogica=especificacion,
                    generador=generador,
                    feedback_revisor=feedback,
                )
                break
            except ErrorSalidaInvalida as exc:
                if llamada >= MAX_REINTENTOS_SALIDA_INVALIDA:
                    return _sin_cambio(exc.mensaje_usuario, exc.mensaje_tecnico)
                log.info("Modificador: salida inválida, reintento interno: %s", exc.mensaje_tecnico)
                feedback["error_de_formato_anterior"] = exc.mensaje_tecnico[:MAX_CHARS_FEEDBACK]
            except (ErrorFormatoNoDisponible, ErrorLLM) as exc:
                return _sin_cambio(exc.mensaje_usuario, exc.mensaje_tecnico)

        if resultado.estado is EstadoRedactor.EVIDENCIA_INSUFICIENTE:
            return _sin_cambio(MENSAJE_SIN_EVIDENCIA, resultado.motivo_abstencion or "")

        salida = {
            **construir_salida_estado(solicitud, resultado.contenido),
            "aviso_modificacion": None,
            "feedback_redactor": None,  # ya consumido
            "intentos_modificador": intentos,
        }
        if chunks_nuevos:
            salida["chunks_fuente_confirmados"] = chunks  # el Revisor los necesita
        return salida

    return nodo_modificador


def enrutar_tras_modificador(state: AgentState) -> str:
    if state.get("aviso_modificacion"):
        # 1ra ejecución: nada cambió -> volver a preguntar. Después: hay una versión
        # rechazada por el Revisor en el state -> guardarla con advertencias.
        return "sin_cambio" if state.get("intentos_modificador", 0) <= 1 else "guardar"
    return "revisar"