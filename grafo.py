"""
grafo.py

Grafo de NuevaMente (5to spec). Flujo secuencial, no fan-out:

buscador_documentos -> confirmar_ejecucion (HITL) -> ingesta -> validacion
-> supervisor -> investigador -> redactor_pedagogico -> critico_revisor
-> guardado_final -> confirmar_modificacion (HITL) -> [modificador -> critico_revisor
   -> guardado_final -> confirmar_modificacion]* -> END

Nodos REALES (llaman a OCI vía Cliente_agemte.py, ya validado):
  - buscador_documentos, ingesta (descarga), guardado_final

Nodos con LLM real:
  - supervisor (agentes/supervisor.py, IntencionOut vía Groq)

Nodos STUB (marcados # TODO, esperando sus archivos en agentes/):
  - investigador, redactor_pedagogico, critico_revisor, modificador

TODO: la rama "aclaracion" de enrutar_tras_investigador hoy corta a END.
Falta un nodo de interrupt propio (1 ronda HITL, sección 2 del 4to spec)
que use el mensaje_aclaracion real que va a generar agente_investigador.py
-- no se puede escribir bien sin ese archivo, queda pendiente a propósito.

Checkpointer: SqliteSaver con conexión manual (aiosqlite.connect()).

IMPORTANTE (evidencia de pruebas repetidas en Windows): con_reintento_mcp
en Cliente_agemte.py NO usa timeout (ni asyncio.wait_for ni
anyio.fail_after) -- envolver la llamada en cualquier cancel scope
provoca cuelgues consistentes del subproceso MCP en el ProactorEventLoop
de Windows. El cuelgue intermitente que queda (sin wrapper) se resuelve
reintentando la corrida manualmente; no bloquea el desarrollo, y es
poco probable que aparezca igual en producción (Linux).
"""

from __future__ import annotations

import json
import re

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent_state import AgentState
from agentes.supervisor import construir_nodo_supervisor
from Cliente_agemte import conectar_mcp, con_reintento_mcp, extraer_texto_resultado, extraer_lista_resultado, obtener_tools_langchain
from seguridad.rate_limiter import RateLimiter

PREFIJO_FUENTES = "fuentes/"
MAX_REINTENTOS_REDACTOR = 2
MAX_VUELTAS_MODIFICACION = 5


# --------------------------------------------------------------------------
# Buscador de Documentos (REAL, determinístico -- 5to spec)
# --------------------------------------------------------------------------

def _matchear_documentos(tema: str, documentos: list[dict]) -> list[str]:
    """
    Match determinístico por nombre de archivo, sin LLM (mismo principio
    de "acceso a OCI no agéntico" ya confirmado para Ingesta/Guardado).
    Devuelve la lista de nombres (sin prefijo) que matchean.
    """
    tema_norm = tema.lower().strip()
    palabras_tema = {p for p in re.split(r"\W+", tema_norm) if len(p) > 2}

    candidatos = []
    for doc in documentos:
        nombre_norm = doc["nombre"].lower()
        if tema_norm in nombre_norm or any(p in nombre_norm for p in palabras_tema):
            candidatos.append(doc["nombre"])
    return candidatos


async def nodo_buscador_documentos(state: AgentState) -> dict:
    tema = state.get("tema_pedido_chat", "")

    async def _listar():
        async with conectar_mcp() as sesion:
            tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
            resultado = await tools["listar_documentos_fuente"].ainvoke({})
            return extraer_lista_resultado(resultado)

    documentos = await con_reintento_mcp(_listar)
    candidatos = _matchear_documentos(tema, documentos)

    if len(candidatos) == 1:
        return {
            "objeto_id_confirmado": f"{PREFIJO_FUENTES}{candidatos[0]}",
            "candidatos_documento": [],
        }

    return {
        "objeto_id_confirmado": None,
        "candidatos_documento": candidatos if candidatos else [d["nombre"] for d in documentos],
    }


# --------------------------------------------------------------------------
# HITL: ¿Ejecutar? (REAL -- 5to spec)
# --------------------------------------------------------------------------

def nodo_confirmar_ejecucion(state: AgentState) -> dict:
    """
    Pausa el grafo hasta que la UI resuma con Command(resume=...).
    Se espera un dict como el de vuelta: {"confirmado": bool, "objeto_id_confirmado": str | None}
    -- si la UI dejó elegir un candidato del selectbox, viene acá.
    """
    respuesta = interrupt({
        "tipo": "confirmar_ejecucion",
        "candidatos_documento": state.get("candidatos_documento", []),
        "objeto_id_confirmado": state.get("objeto_id_confirmado"),
    })

    objeto_id = respuesta.get("objeto_id_confirmado") or state.get("objeto_id_confirmado")
    if objeto_id and not objeto_id.startswith(PREFIJO_FUENTES):
        objeto_id = f"{PREFIJO_FUENTES}{objeto_id}"

    return {
        "ejecucion_confirmada": bool(respuesta.get("confirmado", False)),
        "objeto_id_confirmado": objeto_id,
    }


def enrutar_tras_confirmar_ejecucion(state: AgentState) -> str:
    return "ingesta" if state.get("ejecucion_confirmada") else "cancelado"


# --------------------------------------------------------------------------
# Ingesta (descarga REAL vía Cliente_agemte; chunking/Chroma -- TODO, rag/ pendiente)
# --------------------------------------------------------------------------

async def nodo_ingesta(state: AgentState) -> dict:
    objeto_id = state["objeto_id_confirmado"]

    async def _descargar():
        async with conectar_mcp() as sesion:
            tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
            resultado = await tools["descargar_documento"].ainvoke({"objeto_id": objeto_id})
            texto = extraer_texto_resultado(resultado)
            return json.loads(texto)

    documento = await con_reintento_mcp(_descargar)

    # TODO (rag/ pendiente): chunking + embeddings (HuggingFace multilingual-e5-base)
    # + indexado en Chroma (colección documento_activo, umbral 0.78).
    # Por ahora el texto_extraido queda descargado pero sin indexar; el
    # Investigador stub no lo usa todavía.

    return {"estado": "ingesta_completada"}


# --------------------------------------------------------------------------
# Validación (STUB -- seguridad/validadores.py pendiente)
# --------------------------------------------------------------------------

def nodo_validacion(state: AgentState) -> dict:
    # TODO: sanitización + heurística de prompt injection real
    # (seguridad/validadores.py pendiente de escribir para este proyecto).
    return {"input_sanitizado": True}


def enrutar_tras_validacion(state: AgentState) -> str:
    return "ok"  # TODO: "rechazado" cuando validadores.py esté escrito


# --------------------------------------------------------------------------
# Investigador RAG (STUB)
# --------------------------------------------------------------------------

def nodo_investigador(state: AgentState) -> dict:
    # TODO: consulta real a Chroma (rag/vectorstore.py pendiente).
    # Stub siempre "confirma" para que el skeleton fluya de punta a punta.
    return {
        "fuente_confirmada": True,
        "chunks_fuente_confirmados": ["[STUB] chunk de ejemplo del documento"],
    }


def enrutar_tras_investigador(state: AgentState) -> str:
    # TODO: "aclaracion" hoy corta a END -- falta el nodo de interrupt
    # propio (ver nota al inicio del archivo).
    return "match" if state.get("fuente_confirmada") else "aclaracion"


# --------------------------------------------------------------------------
# Redactor Pedagógico (STUB)
# --------------------------------------------------------------------------

def nodo_redactor_pedagogico(state: AgentState) -> dict:
    # TODO: generación real vía LLM (schema por formato_salida, sección 7/10).
    intentos = state.get("intentos_redactor", 0) + 1
    return {
        "contenido_adaptado": {
            "titulo": "[STUB] Título de ejemplo",
            "introduccion_contextualizada": "[STUB]",
            "items": [],
        },
        "metadatos": {"perfil_aplicado": state.get("perfil_destinatario", "")},
        "intentos_redactor": intentos,
    }


# --------------------------------------------------------------------------
# Crítico/Revisor (STUB, reusado para Redactor y Modificador)
# --------------------------------------------------------------------------

def nodo_critico_revisor(state: AgentState) -> dict:
    # TODO: revisión real vía LLM (formato + fidelidad, sección 10).
    # Stub siempre aprueba para que el skeleton fluya de punta a punta.
    return {
        "aprobado": True,
        "evaluacion_calidad": {"claridad_pedagogica": "Alta", "observaciones": "[STUB]"},
    }


def enrutar_tras_revisor(state: AgentState) -> str:
    if state.get("aprobado"):
        return "guardado_final"
    if state.get("intentos_redactor", 0) >= MAX_REINTENTOS_REDACTOR:
        return "guardado_final"  # agota reintentos -> exito_con_advertencias
    if state.get("vueltas_modificacion", 0) > 0:
        return "modificador"
    return "redactor_pedagogico"


# --------------------------------------------------------------------------
# Guardado final (REAL -- usa guardar_resultado_formateado, ya validado)
# --------------------------------------------------------------------------

async def nodo_guardado_final(state: AgentState) -> dict:
    nombre_archivo = state["objeto_id_confirmado"].replace(PREFIJO_FUENTES, "", 1)
    contenido = {
        "status": "exito" if state.get("intentos_redactor", 0) < MAX_REINTENTOS_REDACTOR else "exito_con_advertencias",
        "metadatos": state.get("metadatos", {}),
        "contenido_adaptado": state.get("contenido_adaptado", {}),
        "evaluacion_calidad": state.get("evaluacion_calidad", {}),
    }

    async def _guardar():
        async with conectar_mcp() as sesion:
            tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
            resultado = await tools["guardar_resultado_formateado"].ainvoke({
                "nombre_archivo": nombre_archivo,
                "contenido": contenido,
            })
            texto = extraer_texto_resultado(resultado)
            return json.loads(texto)

    almacenamiento_oci = await con_reintento_mcp(_guardar)

    return {
        "almacenamiento_oci": almacenamiento_oci,
        "status": contenido["status"],
    }


# --------------------------------------------------------------------------
# HITL: ¿Modificar? (REAL)
# --------------------------------------------------------------------------

def enrutar_tras_guardado(state: AgentState) -> str:
    if state.get("vueltas_modificacion", 0) >= MAX_VUELTAS_MODIFICACION:
        return "tope_alcanzado"
    return "preguntar"


def nodo_confirmar_modificacion(state: AgentState) -> dict:
    respuesta = interrupt({
        "tipo": "confirmar_modificacion",
        "vueltas_modificacion": state.get("vueltas_modificacion", 0),
    })
    return {
        "instruccion_modificacion": respuesta.get("instruccion"),
    }


def enrutar_tras_confirmar_modificacion(state: AgentState) -> str:
    return "modificar" if state.get("instruccion_modificacion") else "fin"


# --------------------------------------------------------------------------
# Modificador (STUB)
# --------------------------------------------------------------------------

def nodo_modificador(state: AgentState) -> dict:
    # TODO: aplica el cambio real vía LLM sobre contenido_adaptado,
    # mismo schema y reglas de fidelidad que el Redactor (sección 12).
    vueltas = state.get("vueltas_modificacion", 0) + 1
    return {
        "contenido_adaptado": {
            **state.get("contenido_adaptado", {}),
            "titulo": f"[STUB modificado v{vueltas}]",
        },
        "vueltas_modificacion": vueltas,
    }


# --------------------------------------------------------------------------
# Construcción del grafo
# --------------------------------------------------------------------------

async def construir_grafo(rate_limiter: RateLimiter | None = None):
    """
    rate_limiter: compartido entre los nodos con LLM real (hoy solo
    Supervisor; se suma Investigador/Redactor/Revisor/Modificador a
    medida que se escriben). Si no se pasa, se crea uno nuevo por
    default -- útil para scripts de prueba sueltos, pero en app.py
    conviene crear UNA instancia y reusarla entre reruns.
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    builder = StateGraph(AgentState)

    builder.add_node("buscador_documentos", nodo_buscador_documentos)
    builder.add_node("confirmar_ejecucion", nodo_confirmar_ejecucion)
    builder.add_node("ingesta", nodo_ingesta)
    builder.add_node("validacion", nodo_validacion)
    builder.add_node("supervisor", construir_nodo_supervisor(rate_limiter))
    builder.add_node("investigador", nodo_investigador)
    builder.add_node("redactor_pedagogico", nodo_redactor_pedagogico)
    builder.add_node("critico_revisor", nodo_critico_revisor)
    builder.add_node("guardado_final", nodo_guardado_final)
    builder.add_node("confirmar_modificacion", nodo_confirmar_modificacion)
    builder.add_node("modificador", nodo_modificador)

    builder.set_entry_point("buscador_documentos")

    builder.add_edge("buscador_documentos", "confirmar_ejecucion")
    builder.add_conditional_edges(
        "confirmar_ejecucion", enrutar_tras_confirmar_ejecucion,
        {"ingesta": "ingesta", "cancelado": END},
    )
    builder.add_edge("ingesta", "validacion")
    builder.add_conditional_edges(
        "validacion", enrutar_tras_validacion, {"rechazado": END, "ok": "supervisor"},
    )
    builder.add_edge("supervisor", "investigador")
    builder.add_conditional_edges(
        "investigador", enrutar_tras_investigador,
        {"match": "redactor_pedagogico", "aclaracion": END},
    )
    builder.add_edge("redactor_pedagogico", "critico_revisor")
    builder.add_conditional_edges(
        "critico_revisor", enrutar_tras_revisor,
        {
            "guardado_final": "guardado_final",
            "redactor_pedagogico": "redactor_pedagogico",
            "modificador": "modificador",
        },
    )
    builder.add_edge("modificador", "critico_revisor")
    builder.add_conditional_edges(
        "guardado_final", enrutar_tras_guardado,
        {"preguntar": "confirmar_modificacion", "tope_alcanzado": END},
    )
    builder.add_conditional_edges(
        "confirmar_modificacion", enrutar_tras_confirmar_modificacion,
        {"modificar": "modificador", "fin": END},
    )

    conexion = await aiosqlite.connect("checkpoints.sqlite")
    checkpointer = AsyncSqliteSaver(conexion)

    return builder.compile(checkpointer=checkpointer)