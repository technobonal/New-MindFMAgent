"""
grafo.py

Grafo de NuevaMente (5to spec). Flujo secuencial, no fan-out:

buscador_documentos -> confirmar_ejecucion (HITL) -> ingesta -> validacion
-> supervisor -> investigador -> [confirmar_aclaracion (HITL, 1 ronda) -> investigador]
-> redactor_pedagogico (-> reintento propio | abstención -> END) -> critico_revisor
-> guardado_final -> confirmar_modificacion (HITL) -> [modificador -> critico_revisor
   -> guardado_final -> confirmar_modificacion]* -> END

Nodos REALES (llaman a OCI vía Cliente_agemte.py, ya validado):
  - buscador_documentos, ingesta (descarga), guardado_final

Nodos con LLM real:
  - supervisor (agentes/supervisor.py, IntencionOut vía Groq)
  - investigador (agentes/agente_investigador.py)
  - redactor_pedagogico (agentes/agente_redactor_pedagogico.py, core del equipo)
  - critico_revisor (agentes/agente_critico_revisor.py: anclaje por coseno + 1 llamada LLM)
  - modificador (agentes/agente_modificador.py, reusa el core del Redactor)

Nodos STUB: ninguno (quedan pendientes validadores.py y guardia_red.py).

RAG: construir_grafo() recibe `buscar_chunks`, `indexar_documento` y
`calcular_anclaje` (vienen de rag/vectorstore.py). Si no se pasan, se usan
stubs que mantienen el skeleton fluyendo de punta a punta. Todas reciben
user_id y objeto_id para aislar cada documento/usuario dentro de Chroma
(ver tipos más abajo).

Contadores (quién los lleva):
  - intentos_redactor:      lo incrementa el Redactor (tope MAX_INTENTOS_REDACTOR).
  - vueltas_modificacion:   lo incrementa el GRAFO en confirmar_modificacion, una
                            vez por instrucción del usuario (tope de sesión = 5).
  - intentos_modificador:   lo incrementa el Modificador en cada ejecución; el
                            grafo lo resetea a 0 con cada instrucción nueva
                            (tope MAX_INTENTOS_MODIFICADOR por instrucción).

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

import asyncio
import json
import logging
import re
from pathlib import PurePosixPath
from typing import Callable

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent_state import AgentState
from agentes.agente_critico_revisor import CalcularAnclaje, construir_nodo_revisor
from agentes.agente_investigador import construir_nodo_investigador
from agentes.agente_modificador import construir_nodo_modificador, enrutar_tras_modificador
from agentes.agente_redactor_pedagogico import construir_nodo_redactor, enrutar_tras_redactor
from agentes.supervisor import construir_nodo_supervisor
from Cliente_agemte import conectar_mcp, con_reintento_mcp, extraer_texto_resultado, extraer_lista_resultado, obtener_tools_langchain
from seguridad.rate_limiter import RateLimiter
from src.errores import MENSAJE_POR_DEFECTO, CodigoError

log = logging.getLogger(__name__)

PREFIJO_FUENTES = "fuentes/"
MAX_INTENTOS_REDACTOR = 3        # 1 intento inicial + 2 correcciones
MAX_INTENTOS_MODIFICADOR = 3     # por cada instrucción de modificación del usuario
MAX_VUELTAS_MODIFICACION = 5     # instrucciones de modificación por sesión

# (texto, objeto_id, user_id) -> None.
# Chunking + embeddings + indexado en Chroma. Debe ser IDEMPOTENTE: si ese
# (user_id, objeto_id) ya está indexado con el mismo contenido, no reindexa.
IndexarDocumento = Callable[[str, str, str], None]

# (consulta, k, user_id, objeto_id) -> [(id_chunk, texto_chunk, similitud_coseno), ...] desc.
# Debe filtrar SOLO los chunks de ese user_id + objeto_id. El umbral (0.78)
# NO se aplica acá: lo aplica el Investigador. El id_chunk es el ID real de
# Chroma: el Redactor lo usa en el `anclaje` de cada item.
BuscarChunksEnAlcance = Callable[[str, int, str, str], list[tuple[str, str, float]]]

# CalcularAnclaje (afirmaciones, user_id, objeto_id) -> dict: ver agente_critico_revisor.py


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _user_id(config: RunnableConfig) -> str:
    """
    El thread_id del checkpointer es el email del usuario (st.user.email),
    así que sirve como identificador de aislamiento sin sumar campos al estado.
    """
    return str(config["configurable"]["thread_id"])


def _nombre_versionado(nombre_archivo: str, version: int) -> str:
    """'manual.pdf' + v2 -> 'manual_v2.pdf' (no sobreescribe versiones previas)."""
    p = PurePosixPath(nombre_archivo)
    return str(p.with_name(f"{p.stem}_v{version}{p.suffix}"))


# --------------------------------------------------------------------------
# Stubs de RAG (para scripts de prueba sin rag/vectorstore.py)
# --------------------------------------------------------------------------

def _buscar_chunks_stub(consulta: str, k: int, user_id: str, objeto_id: str) -> list[tuple[str, str, float]]:
    return [("stub-0", "[STUB] chunk de ejemplo del documento", 1.0)]


def _indexar_documento_stub(texto: str, objeto_id: str, user_id: str) -> None:
    return None


def _calcular_anclaje_stub(afirmaciones: list[str], user_id: str, objeto_id: str) -> dict:
    return {
        "score": 1.0,
        "minimo": 1.0,
        "por_afirmacion": [{"afirmacion": a, "similitud": 1.0} for a in afirmaciones],
    }


# --------------------------------------------------------------------------
# Buscador de Documentos (REAL, determinístico -- 5to spec)
# --------------------------------------------------------------------------

# Campos por-pedido que deben arrancar limpios en cada corrida: el thread_id
# (email del usuario) persiste el estado entre pedidos en el checkpointer.
RESET_PEDIDO = {
    # aclaración del Investigador
    "mensaje_aclaracion": None,
    "respuesta_aclaracion_usuario": None,
    "fuente_confirmada": None,
    "chunks_fuente_confirmados": [],
    # confirmación de ejecución
    "ejecucion_confirmada": False,
    # generación y revisión
    "contenido_adaptado": {},
    "metadatos": {},
    "evaluacion_calidad": {},
    "aprobado": None,
    "intentos_redactor": 0,
    "feedback_redactor": None,
    # modificación
    "instruccion_modificacion": None,
    "vueltas_modificacion": 0,
    "intentos_modificador": 0,
    "aviso_modificacion": None,
    # resultado
    "status": None,
    "error": None,
    "almacenamiento_oci": None,
}


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
            **RESET_PEDIDO,
            "objeto_id_confirmado": f"{PREFIJO_FUENTES}{candidatos[0]}",
            "candidatos_documento": [],
        }

    return {
        **RESET_PEDIDO,
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
        # Sin documento elegido no se puede ejecutar, aunque la UI mande confirmado=True.
        "ejecucion_confirmada": bool(respuesta.get("confirmado", False)) and bool(objeto_id),
        "objeto_id_confirmado": objeto_id,
    }


def enrutar_tras_confirmar_ejecucion(state: AgentState) -> str:
    return "ingesta" if state.get("ejecucion_confirmada") else "cancelado"


# --------------------------------------------------------------------------
# Ingesta (descarga REAL vía Cliente_agemte; indexado vía rag/ inyectado)
# --------------------------------------------------------------------------

def construir_nodo_ingesta(indexar_documento: IndexarDocumento):
    async def nodo_ingesta(state: AgentState, config: RunnableConfig) -> dict:
        objeto_id = state["objeto_id_confirmado"]
        user_id = _user_id(config)

        async def _descargar():
            async with conectar_mcp() as sesion:
                tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
                resultado = await tools["descargar_documento"].ainvoke({"objeto_id": objeto_id})
                texto = extraer_texto_resultado(resultado)
                return json.loads(texto)

        documento = await con_reintento_mcp(_descargar)

        # Chunking + embeddings (HF remoto, bloqueante) + Chroma, fuera del
        # event loop. El texto va directo a Chroma, no al state, para no
        # inflar el checkpoint de SQLite.
        await asyncio.to_thread(indexar_documento, documento["texto_extraido"], objeto_id, user_id)

        return {"estado": "ingesta_completada"}

    return nodo_ingesta


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
# Investigador RAG (REAL -- agentes/agente_investigador.py) + HITL aclaración
# --------------------------------------------------------------------------

def construir_nodo_investigador_con_alcance(rate_limiter: RateLimiter, buscar_chunks: BuscarChunksEnAlcance):
    """
    Envuelve al Investigador para que su búsqueda quede acotada al documento y
    usuario actuales, sin tocar agente_investigador.py (él sigue llamando a
    buscar_chunks(consulta, k)).
    """
    def nodo_investigador(state: AgentState, config: RunnableConfig) -> dict:
        user_id = _user_id(config)
        objeto_id = state["objeto_id_confirmado"]

        def _buscar_en_alcance(consulta: str, k: int) -> list[tuple[str, str, float]]:
            return buscar_chunks(consulta, k, user_id, objeto_id)

        return construir_nodo_investigador(rate_limiter, _buscar_en_alcance)(state)

    return nodo_investigador


def enrutar_tras_investigador(state: AgentState) -> str:
    if state.get("fuente_confirmada"):
        return "match"
    if state.get("mensaje_aclaracion"):
        return "aclaracion"
    return "sin_fuente"  # tope de 1 ronda agotado -> status "error"


def nodo_confirmar_aclaracion(state: AgentState) -> dict:
    respuesta = interrupt({
        "tipo": "confirmar_aclaracion",
        "mensaje_aclaracion": state.get("mensaje_aclaracion"),
    })
    texto = (respuesta.get("respuesta") or "").strip()
    return {"respuesta_aclaracion_usuario": texto or None}


def enrutar_tras_aclaracion(state: AgentState) -> str:
    return "reintentar" if state.get("respuesta_aclaracion_usuario") else "cancelado"


# --------------------------------------------------------------------------
# Redactor Pedagógico: REAL (agentes/agente_redactor_pedagogico.py).
# Su router (enrutar_tras_redactor) se importa arriba.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Crítico/Revisor: REAL (agentes/agente_critico_revisor.py), reusado para
# Redactor y Modificador. Aquí solo vive su router.
# --------------------------------------------------------------------------

def enrutar_tras_revisor(state: AgentState) -> str:
    if state.get("aprobado"):
        return "guardado_final"

    # Vuelta de modificación: el rechazo vuelve al Modificador, con su propio tope.
    if state.get("vueltas_modificacion", 0) > 0:
        if state.get("intentos_modificador", 0) >= MAX_INTENTOS_MODIFICADOR:
            return "guardado_final"  # agota correcciones -> exito_con_advertencias
        return "modificador"

    # Primera generación: el rechazo vuelve al Redactor.
    if state.get("intentos_redactor", 0) >= MAX_INTENTOS_REDACTOR:
        return "guardado_final"  # agota reintentos -> exito_con_advertencias
    return "redactor_pedagogico"


# --------------------------------------------------------------------------
# Guardado final (REAL -- usa guardar_resultado_formateado, ya validado)
# --------------------------------------------------------------------------

async def nodo_guardado_final(state: AgentState) -> dict:
    nombre_original = state["objeto_id_confirmado"].replace(PREFIJO_FUENTES, "", 1)

    # v1 = generación original; cada modificación del usuario suma una versión.
    version = state.get("vueltas_modificacion", 0) + 1
    nombre_archivo = _nombre_versionado(nombre_original, version)

    # El estado final depende de si el Revisor aprobó, no de cuántos intentos hubo.
    status = "exito" if state.get("aprobado") else "exito_con_advertencias"

    contenido = {
        "status": status,
        "version": version,
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

    # Decisión D-05: un fallo de carga a OCI NO aborta. El contenido sigue en el
    # state (la UI puede mostrarlo y ofrecer la descarga) y el paquete sale con
    # status="exito_con_advertencias" y status_upload="fallido".
    try:
        almacenamiento_oci = await con_reintento_mcp(_guardar)
        guardado_ok = not (
            isinstance(almacenamiento_oci, dict)
            and str(almacenamiento_oci.get("status_upload", "")).lower() == "fallido"
        )
    except Exception as exc:  # incluye ExceptionGroup de anyio y JSON inválido
        log.warning("guardado_final: falló la carga a OCI: %s", exc)
        almacenamiento_oci = {
            "bucket": "",
            "objeto_id": f"generados/formateados/{nombre_archivo}",
            "status_upload": "fallido",
            "detalle_fallo": MENSAJE_POR_DEFECTO[CodigoError.FALLO_PERSISTENCIA],
        }
        guardado_ok = False

    if not guardado_ok:
        status = "exito_con_advertencias"

    return {
        "almacenamiento_oci": almacenamiento_oci,
        "status": status,
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
        "aviso": state.get("aviso_modificacion"),  # p.ej. "no se pudo aplicar ese cambio"
    })
    instruccion = (respuesta.get("instruccion") or "").strip() or None

    if not instruccion:
        return {"instruccion_modificacion": None, "aviso_modificacion": None}

    # Cada instrucción nueva: cuenta como 1 vuelta de sesión y arranca con
    # el contador de correcciones del Modificador en cero.
    return {
        "instruccion_modificacion": instruccion,
        "vueltas_modificacion": state.get("vueltas_modificacion", 0) + 1,
        "intentos_modificador": 0,
        "aviso_modificacion": None,
        "aprobado": None,
    }


def enrutar_tras_confirmar_modificacion(state: AgentState) -> str:
    return "modificar" if state.get("instruccion_modificacion") else "fin"


# --------------------------------------------------------------------------
# Modificador: REAL (agentes/agente_modificador.py). Su router se importa arriba.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Construcción del grafo
# --------------------------------------------------------------------------

async def construir_grafo(
    rate_limiter: RateLimiter | None = None,
    buscar_chunks: BuscarChunksEnAlcance | None = None,
    indexar_documento: IndexarDocumento | None = None,
    calcular_anclaje: CalcularAnclaje | None = None,
):
    """
    rate_limiter: compartido entre los nodos con LLM real (Supervisor,
    Investigador, Redactor, Revisor y Modificador). Si no se pasa, se crea uno nuevo por default -- útil para
    scripts de prueba sueltos, pero en app.py conviene crear UNA instancia
    y reusarla entre reruns.

    buscar_chunks / indexar_documento / calcular_anclaje: vienen de
    rag/vectorstore.py (buscar_chunks_con_id, indexar_documento,
    calcular_anclaje), con las firmas
    BuscarChunksEnAlcance e IndexarDocumento (reciben user_id y objeto_id).
    Si no se pasan, se usan stubs.
    """
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    if buscar_chunks is None:
        buscar_chunks = _buscar_chunks_stub
    if indexar_documento is None:
        indexar_documento = _indexar_documento_stub
    if calcular_anclaje is None:
        calcular_anclaje = _calcular_anclaje_stub

    builder = StateGraph(AgentState)

    builder.add_node("buscador_documentos", nodo_buscador_documentos)
    builder.add_node("confirmar_ejecucion", nodo_confirmar_ejecucion)
    builder.add_node("ingesta", construir_nodo_ingesta(indexar_documento))
    builder.add_node("validacion", nodo_validacion)
    builder.add_node("supervisor", construir_nodo_supervisor(rate_limiter))
    builder.add_node("investigador", construir_nodo_investigador_con_alcance(rate_limiter, buscar_chunks))
    builder.add_node("confirmar_aclaracion", nodo_confirmar_aclaracion)
    builder.add_node("redactor_pedagogico", construir_nodo_redactor(rate_limiter, MAX_INTENTOS_REDACTOR))
    builder.add_node(
        "critico_revisor",
        construir_nodo_revisor(
            rate_limiter, calcular_anclaje, MAX_INTENTOS_REDACTOR, MAX_INTENTOS_MODIFICADOR
        ),
    )
    builder.add_node("guardado_final", nodo_guardado_final)
    builder.add_node("confirmar_modificacion", nodo_confirmar_modificacion)
    builder.add_node(
        "modificador",
        construir_nodo_modificador(rate_limiter, buscar_chunks),
    )

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
        {"match": "redactor_pedagogico", "aclaracion": "confirmar_aclaracion", "sin_fuente": END},
    )
    builder.add_conditional_edges(
        "confirmar_aclaracion", enrutar_tras_aclaracion,
        {"reintentar": "investigador", "cancelado": END},
    )
    builder.add_conditional_edges(
        "redactor_pedagogico", enrutar_tras_redactor,
        {"revisar": "critico_revisor", "reintentar": "redactor_pedagogico", "abstencion": END},
    )
    builder.add_conditional_edges(
        "critico_revisor", enrutar_tras_revisor,
        {
            "guardado_final": "guardado_final",
            "redactor_pedagogico": "redactor_pedagogico",
            "modificador": "modificador",
        },
    )
    builder.add_conditional_edges(
        "modificador", enrutar_tras_modificador,
        {"revisar": "critico_revisor", "sin_cambio": "confirmar_modificacion", "guardar": "guardado_final"},
    )
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