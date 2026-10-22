"""
app.py

UI de Streamlit para NuevaMente. Primera versión funcional:
- Input de tema por chat (Buscador de Documentos)
- Uploader para subir un documento nuevo directo a fuentes/ en OCI
  (Flujo B) -- queda disponible para el Buscador en la próxima búsqueda
- Asistente de parámetros paso a paso: el chat pide documento, perfil y formato
  de a uno, con opciones tocables (o texto escrito); salta lo que el mensaje
  inicial ya trae
- Confirmación "¿Ejecutar?" con documento y los 4 parámetros editables
  (nicho y nivel arrancan en General / Estandar)
- HITL "aclaración" y "¿Modificar?" vía interrupt() + Command(resume=...)
- Timeline en vivo de los nodos del grafo vía astream(), con el detalle de
  lo que hizo cada fase y persistente entre reruns
- Salida estructurada (JSON) y render del contenido según el formato
  (Tutorial, Flashcards, Quiz, Resumen Ejecutivo)
- Ficha/reporte con BLUF, hallazgos, recomendaciones, evaluación y
  botones Imprimir / Guardar en OCI / Otra consulta
- Estética: hero navy oscuro con vórtice de fondo, acento cian con glow
- Navegación por pestañas: Inicio | Agentes | Cómo funciona | Casos de uso | Seguridad | Actividad Agentica

Acceso público con login de Google (seguridad/auth.py, st.login): cada usuario se
identifica con su `sub` de Google (user_id), que aísla sus documentos en OCI y en
Chroma. Hay cuotas de corridas por día (seguridad/cuotas.py).

Nota de arquitectura: cada interacción abre una conexión nueva al grafo
(asyncio.run + construir_grafo) y la cierra al terminar.
"""

import asyncio
import base64
import html
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nuevamente.app")

import streamlit as st
from langgraph.types import Command
from rag.vectorstore import buscar_chunks_con_id, indexar_documento, calcular_anclaje

from grafo import construir_grafo
# Nota: el módulo se llama Cliente_agemte (typo histórico); no renombrar el import
# hasta que el archivo fuente se renombre de forma coordinada.
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado
from seguridad.auth import requerir_login
from seguridad.cuotas import ErrorCuota, consumir_corrida, corridas_restantes, limite_usuario
from seguridad.rate_limiter import RateLimiter
from render_documento import CSS_DOCUMENTO, tipo_documento, html_documento

st.set_page_config(page_title="NuevaMente", page_icon="🧠", layout="wide")
st.markdown(f"<style>{CSS_DOCUMENTO}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Constantes de configuración
# --------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB (mismo tope que el servidor MCP: OCI_MAX_UPLOAD_MB)
MAX_TIMELINE_NODOS = 40
TIPOS_INTERRUPT = frozenset({
    "pedir_parametro",
    "confirmar_ejecucion",
    "confirmar_aclaracion",
    "confirmar_modificacion",
})
EXTENSIONES_PERMITIDAS = frozenset({".pdf", ".md", ".txt"})

# --------------------------------------------------------------------------
# Estética
# --------------------------------------------------------------------------

st.markdown("""
<style>
:root {
    --bg: #0A121F;
    --bg-elevated: #131E33;
    --glow-primary: #3EA6FF;
    --glow-secondary: #4FC3F7;
    --text-primary: #F5F7FA;
    --text-secondary: #8B96A8;
    --border-subtle: rgba(62, 166, 255, 0.18);
}

.stApp {
    background:
        radial-gradient(ellipse 900px 700px at 15% 20%, rgba(62, 166, 255, 0.16), transparent 60%),
        var(--bg);
    color: var(--text-primary);
}

.nm-hero {
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 28px;
    background: radial-gradient(ellipse 120% 90% at 25% 55%, #16274A 0%, var(--bg) 68%);
    border: 1px solid var(--border-subtle);
}

.nm-hero::before {
    content: "";
    position: absolute;
    top: 50%;
    left: -10%;
    width: 620px;
    height: 620px;
    transform: translateY(-50%);
    transform-origin: center center;
    border-radius: 50%;
    background:
        repeating-conic-gradient(
            from 0deg,
            rgba(90, 210, 255, 0.30) 0deg 3deg,
            transparent 3deg 22deg
        ),
        repeating-radial-gradient(
            circle at center,
            rgba(90, 210, 255, 0.5) 0px,
            rgba(90, 210, 255, 0.5) 2px,
            transparent 3px,
            transparent 14px
        );
    -webkit-mask-image: radial-gradient(circle at center, black 55%, transparent 78%);
    mask-image: radial-gradient(circle at center, black 55%, transparent 78%);
    filter: blur(0.4px) drop-shadow(0 0 14px rgba(90, 210, 255, 0.55));
    pointer-events: none;
    animation: nm-spin 40s linear infinite;
}

@keyframes nm-spin {
    from { transform: translateY(-50%) rotate(0deg); }
    to   { transform: translateY(-50%) rotate(360deg); }
}

.nm-hero::after {
    content: "";
    position: absolute;
    top: 50%;
    left: -10%;
    width: 620px;
    height: 620px;
    transform: translateY(-50%);
    border-radius: 50%;
    background: radial-gradient(circle at center, transparent 30%, #060B14 32%);
    pointer-events: none;
}

.nm-hero-content {
    position: relative;
    z-index: 1;
    max-width: 640px;
    margin-left: auto;
}

.nm-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--glow-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 4px 12px;
    background: var(--bg-elevated);
}

.nm-badge.completado { color: #4ADE80; }
.nm-badge.pendiente { color: var(--text-secondary); }

.nm-hero-title {
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.01em;
    font-size: clamp(2rem, 4vw, 3.2rem);
    margin: 18px 0 14px 0;
    color: #FFFFFF;
    text-shadow: 0 0 22px rgba(255, 255, 255, 0.35), 0 0 46px rgba(62, 166, 255, 0.25);
}

.nm-hero-title .linea-acento {
    background: linear-gradient(90deg, #7FD8FF, var(--glow-secondary), var(--glow-primary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    display: block;
    text-shadow: none;
    filter: drop-shadow(0 0 18px rgba(79, 195, 247, 0.65)) drop-shadow(0 0 36px rgba(62, 166, 255, 0.4));
}

.nm-hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    max-width: 460px;
}

.nm-nodo {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

.nm-nodo.activo {
    border-color: var(--glow-primary);
    box-shadow: 0 0 24px rgba(62, 166, 255, 0.35), 0 0 4px rgba(62, 166, 255, 0.5);
}

.nm-nodo.en-curso {
    border-color: var(--glow-secondary);
    animation: nm-pulse 1.2s ease-in-out infinite;
}

.nm-nodo.error {
    border-color: #F87171;
    box-shadow: 0 0 18px rgba(248, 113, 113, 0.35);
}

@keyframes nm-pulse {
    0%, 100% { box-shadow: 0 0 6px rgba(79, 195, 247, 0.25); }
    50%      { box-shadow: 0 0 26px rgba(79, 195, 247, 0.60); }
}

.nm-nodo-titulo {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 2px;
}

.nm-rep {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 20px;
}

.nm-rep-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--glow-secondary);
    font-weight: 700;
}

.nm-rep-titulo {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 8px 0 10px 0;
    padding: 0;
    color: #FFFFFF;
}

.nm-rep-meta {
    color: var(--text-secondary);
    font-size: 0.85rem;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border-subtle);
}

.nm-rep-seccion {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-secondary);
    font-weight: 700;
    margin: 28px 0 10px 0;
}

.nm-rep-resumen {
    font-size: 1.05rem;
    line-height: 1.7;
}

.nm-bluf {
    background: rgba(62, 166, 255, 0.10);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0 8px 0;
    font-size: 1.08rem;
    line-height: 1.55;
    font-weight: 600;
    color: #E8F4FF;
}

.nm-hallazgo {
    display: flex;
    gap: 16px;
    padding: 16px 0;
    border-top: 1px solid var(--border-subtle);
}

.nm-hallazgo-num {
    flex: 0 0 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(62, 166, 255, 0.15);
    color: var(--glow-secondary);
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}

.nm-hallazgo-titulo {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 8px;
}

.nm-hallazgo-fila {
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 4px 0;
    color: #D5DBE5;
}

.nm-hallazgo-fila b {
    color: var(--glow-secondary);
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 8px;
}

.nm-hallazgo-fuente {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 6px;
    word-break: break-all;
}

.nm-rep-pie {
    margin-top: 22px;
    padding-top: 14px;
    border-top: 1px solid var(--border-subtle);
    font-size: 0.78rem;
    color: var(--text-secondary);
}

.nm-chip {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--glow-secondary);
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 4px 12px;
    margin: 0 6px 8px 0;
}

.nm-nodo-detalle {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

.nm-reco {
    padding: 8px 0 8px 12px;
    border-left: 3px solid var(--glow-primary);
    margin: 6px 0;
    color: #D5DBE5;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ---------- Navegación por pestañas ---------- */
div[data-baseweb="tab-list"] {
    gap: 8px;
    padding: 8px;
    background: rgba(19, 30, 51, 0.65);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    backdrop-filter: blur(8px);
    width: fit-content;
    max-width: 100%;
    overflow-x: auto;
    margin: 0 auto 28px auto;
}
div[data-baseweb="tab-highlight"] { background: transparent !important; }
div[data-baseweb="tab-border"] { display: none; }

button[data-baseweb="tab"] {
    background: transparent;
    height: auto;
    padding: 14px 28px;
    border-radius: 12px;
    transition: background .2s ease, box-shadow .2s ease;
}
button[data-baseweb="tab"] p {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: #A9B6CA;
    transition: color .2s ease, text-shadow .2s ease;
}
button[data-baseweb="tab"]:hover { background: rgba(62, 166, 255, 0.10); }
button[data-baseweb="tab"]:hover p {
    color: var(--glow-secondary);
    text-shadow: 0 0 14px rgba(79, 195, 247, 0.6);
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(62, 166, 255, 0.16);
    box-shadow:
        0 0 26px rgba(79, 195, 247, 0.45),
        0 0 4px rgba(79, 195, 247, 0.8),
        inset 0 0 0 1px rgba(79, 195, 247, 0.6);
}
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #FFFFFF;
    font-weight: 700;
    text-shadow: 0 0 14px rgba(79, 195, 247, 0.9), 0 0 30px rgba(62, 166, 255, 0.6);
}
@media (max-width: 800px) {
    button[data-baseweb="tab"] { padding: 10px 16px; }
    button[data-baseweb="tab"] p { font-size: 1rem; }
}

/* ---------- Inicio ---------- */
.nm-home-head { text-align: center; max-width: 860px; margin: 28px auto 0 auto; }
.nm-home-title {
    font-weight: 300;
    font-size: clamp(2.2rem, 4.6vw, 3.6rem);
    line-height: 1.15;
    letter-spacing: -0.01em;
    color: #FFFFFF;
    margin: 0 0 20px 0;
    padding: 0;
    text-shadow: 0 0 30px rgba(62, 166, 255, 0.25);
}
.nm-home-title .acento {
    background: linear-gradient(90deg, #7FD8FF, var(--glow-secondary), var(--glow-primary));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 16px rgba(79, 195, 247, 0.55));
}
.nm-home-sub {
    color: var(--text-secondary);
    font-size: 1.1rem;
    line-height: 1.6;
    max-width: 720px;
    margin: 0 auto 28px auto;
}
.nm-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--glow-secondary);
    color: #FFFFFF;
    border-radius: 999px;
    padding: 12px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 0 22px rgba(79, 195, 247, 0.35), inset 0 0 12px rgba(79, 195, 247, 0.12);
}

.nm-mosaic {
    display: grid;
    grid-template-columns: 1.15fr 1.4fr 0.9fr 1.2fr;
    grid-template-rows: 210px 210px;
    gap: 12px;
    margin: 44px 0 40px 0;
}
.nm-tile {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    border: 1px solid var(--border-subtle);
}
.nm-tile svg { width: 100%; height: 100%; display: block; filter: drop-shadow(0 0 8px rgba(79, 195, 247, 0.45)); }
.nm-tile.t1 { grid-row: 1 / 3; background: radial-gradient(ellipse at 50% 85%, #1B3D66 0%, #0C1A2E 70%); }
.nm-tile.t2 { background: linear-gradient(160deg, #15294A, #0C1727); }
.nm-tile.t3 { background: radial-gradient(circle at 50% 50%, #12304F 0%, #0A121F 75%); }
.nm-tile.t4 { background: linear-gradient(200deg, #1A2F55, #0B1525); }
.nm-tile.t5 { background: linear-gradient(140deg, #0F2A3C, #0A121F); }
.nm-tile.t6 { background: radial-gradient(circle at 50% 50%, #17335A 0%, #0A121F 80%); }
.nm-tile.t7 { background: linear-gradient(180deg, #13243F, #0A121F); }

.nm-trio {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 32px;
    padding-top: 32px;
    border-top: 3px solid var(--border-subtle);
    margin-bottom: 24px;
    text-align: center;
}
.nm-trio-titulo { font-size: 1.35rem; font-weight: 500; color: #FFFFFF; margin-bottom: 10px; }
.nm-trio-texto { color: #C5CDD9; font-size: 1rem; line-height: 1.6; margin-bottom: 14px; }
.nm-trio-pista { color: var(--glow-secondary); font-weight: 600; font-size: 0.95rem; text-shadow: 0 0 12px rgba(79, 195, 247, 0.5); }

@media (max-width: 800px) {
    .nm-mosaic { grid-template-columns: 1fr 1fr; grid-template-rows: none; grid-auto-rows: 170px; }
    .nm-tile.t1 { grid-row: auto; }
    .nm-trio { grid-template-columns: 1fr; }
}

/* ---------- Cards ---------- */
.nm-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
    margin: 8px 0 20px 0;
}
.nm-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 22px 22px 18px 22px;
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.nm-card:hover {
    transform: translateY(-3px);
    border-color: var(--glow-primary);
    box-shadow: 0 0 26px rgba(62, 166, 255, 0.28);
}
.nm-card-icono { font-size: 1.6rem; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(79, 195, 247, 0.6)); }
.nm-card-titulo { font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; color: #FFFFFF; }
.nm-card-texto { font-size: 0.9rem; line-height: 1.55; color: #C5CDD9; }
.nm-card-pie {
    margin-top: 12px;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--glow-secondary);
    font-weight: 700;
}

/* ---------- Flujo (Cómo funciona) ---------- */
.nm-flow { position: relative; margin: 8px 0 24px 0; padding-left: 8px; }
.nm-flow::before {
    content: "";
    position: absolute;
    left: 31px; top: 20px; bottom: 20px;
    width: 2px;
    background: linear-gradient(180deg, var(--glow-secondary), rgba(62, 166, 255, 0.1));
}
.nm-step { position: relative; display: flex; gap: 18px; padding: 10px 0; }
.nm-step-num {
    flex: 0 0 46px; height: 46px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--glow-secondary);
    z-index: 1;
}
.nm-step.humano .nm-step-num {
    border-color: var(--glow-secondary);
    box-shadow: 0 0 18px rgba(79, 195, 247, 0.5);
    color: #FFFFFF;
}
.nm-step-titulo { font-weight: 700; font-size: 1.02rem; }
.nm-step-texto { font-size: 0.9rem; color: #C5CDD9; line-height: 1.5; margin: 2px 0 6px 0; }
.nm-step-humano, .nm-step-auto {
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;
    border-radius: 999px; padding: 2px 10px; border: 1px solid var(--border-subtle);
}
.nm-step-humano { color: var(--glow-secondary); background: rgba(62, 166, 255, 0.12); }
.nm-step-auto { color: var(--text-secondary); }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Acceso: login con Google. Sin sesión válida la página se corta aquí.
# --------------------------------------------------------------------------

USUARIO = requerir_login()

with st.sidebar:
    st.markdown("**Tu cuenta**")
    st.caption(USUARIO.nombre or USUARIO.email)
    if USUARIO.nombre:
        st.caption(USUARIO.email)
    st.caption(
        f"Corridas disponibles hoy: {corridas_restantes(USUARIO.user_id)} de {limite_usuario()}"
    )
    st.button("Cerrar sesión", on_click=st.logout, use_container_width=True)

# --------------------------------------------------------------------------
# Helpers async
# --------------------------------------------------------------------------

NOMBRES_NODOS = {
    "buscador_documentos": "Buscador de Documentos",
    "parametros": "Parámetros",
    "confirmar_ejecucion": "Confirmación de ejecución",
    "ingesta": "Ingesta",
    "validacion": "Validación",
    "supervisor": "Supervisor",
    "investigador": "Investigador RAG",
    "confirmar_aclaracion": "Aclaración del tema",
    "redactor_pedagogico": "Redactor Pedagógico",
    "critico_revisor": "Crítico / Revisor",
    "ilustrador": "Ilustrador",
    "guardado_final": "Guardado final",
    "confirmar_modificacion": "Confirmación de modificación",
    "modificador": "Modificador",
    "_error": "Error del pipeline",
}


async def _subir_documento(nombre_archivo: str, contenido_bytes: bytes, user_id: str) -> dict:
    contenido_b64 = base64.b64encode(contenido_bytes).decode("ascii")
    async with conectar_mcp() as sesion:
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        resultado = await tools["subir_documento_fuente"].ainvoke({
            "nombre_archivo": nombre_archivo,
            "contenido_base64": contenido_b64,
            "user_id": user_id,
        })

        # Debug temporal (sácalo cuando funcione)
        print("=== RAW subir_documento_fuente ===", flush=True)
        print(repr(resultado), flush=True)
        print("=================================", flush=True)

        texto = extraer_texto_resultado(resultado)
        if not texto or not texto.strip():
            raise ValueError(f"Tool devolvió vacío. Raw: {resultado!r}")

        parseado = json.loads(texto) if isinstance(texto, str) else texto
        if not isinstance(parseado, dict):
            raise ValueError(f"Se esperaba dict, llegó: {type(parseado)}")
        return parseado


async def _subir_reporte_generado(nombre_archivo: str, contenido_html: str, user_id: str) -> dict:
    contenido_b64 = base64.b64encode(contenido_html.encode("utf-8")).decode("ascii")
    async with conectar_mcp() as sesion:
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        tool = tools.get("subir_documento_fuente") or tools.get("subir_objeto")
        if not tool:
            raise RuntimeError("No se encontró tool de subida en MCP")
        resultado = await tool.ainvoke({
            "nombre_archivo": nombre_archivo,
            "contenido_base64": contenido_b64,
            "user_id": user_id,
        })
        texto = extraer_texto_resultado(resultado)
        return json.loads(texto)


async def _construir_grafo_app():
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    return await construir_grafo(
        rate_limiter=st.session_state.rate_limiter,
        buscar_chunks=buscar_chunks_con_id,
        indexar_documento=indexar_documento,
        calcular_anclaje=calcular_anclaje,
    )


@asynccontextmanager
async def _grafo_sesion():
    """Abre el grafo y garantiza el cierre del checkpointer al salir."""
    grafo = await _construir_grafo_app()
    try:
        yield grafo
    finally:
        try:
            conn = getattr(getattr(grafo, "checkpointer", None), "conn", None)
            if conn is not None:
                await conn.close()
        except Exception as exc:
            logger.warning("No se pudo cerrar checkpointer: %s", exc)


async def _consumir_stream(grafo, entrada, config, contenedor) -> None:
    """Consume astream del grafo y pinta el timeline. Captura errores de nodo."""
    slots: dict[str, Any] = {}
    try:
        async for modo, chunk in grafo.astream(
            entrada, config=config, stream_mode=["debug", "updates"]
        ):
            if modo == "debug":
                if chunk.get("type") == "task":
                    nombre = (chunk.get("payload") or {}).get("name")
                    if nombre in NOMBRES_NODOS:
                        with contenedor:
                            slot = st.empty()
                        slots[nombre] = slot
                        slot.markdown(_html_nodo_en_curso(nombre), unsafe_allow_html=True)
            elif modo == "updates":
                for nombre, datos in chunk.items():
                    _pintar_nodo(
                        contenedor, nombre, activo=True, datos=datos,
                        slot=slots.pop(nombre, None),
                    )
    except Exception as exc:
        logger.exception("Error durante astream del grafo")
        mensaje = getattr(exc, "mensaje_usuario", None) or str(exc) or type(exc).__name__
        _pintar_nodo(
            contenedor,
            "_error",
            activo=True,
            datos={"error": mensaje, "status": "error"},
            slot=slots.pop("_error", None),
        )
        raise
    finally:
        for slot in slots.values():
            try:
                slot.empty()
            except Exception:
                pass


async def _iniciar_flujo(tema: str, thread_id: str, user_id: str, contenedor_timeline):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    estado_inicial = {
        "mensajes": [],
        "thread_id": thread_id,
        "tema_pedido_chat": tema.strip()[:500],
        "candidatos_documento": [],
        "objeto_id_confirmado": None,
        "ejecucion_confirmada": False,
        "intentos_redactor": 0,
        "vueltas_modificacion": 0,
        "status": "",
    }
    async with _grafo_sesion() as grafo:
        await _consumir_stream(grafo, estado_inicial, config, contenedor_timeline)
        return await grafo.aget_state(config)


async def _resumir_flujo(resume_payload: dict, thread_id: str, user_id: str, contenedor_timeline):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    async with _grafo_sesion() as grafo:
        await _consumir_stream(
            grafo, Command(resume=resume_payload), config, contenedor_timeline
        )
        return await grafo.aget_state(config)


def _resumen_nodo(datos) -> list[str]:
    if not isinstance(datos, dict) or not datos:
        return ["Paso ejecutado"]
    lineas = []
    if "candidatos_documento" in datos:
        n = len(datos["candidatos_documento"] or [])
        lineas.append(f"{n} documento(s) candidato(s) encontrado(s)")
    if datos.get("objeto_id_confirmado"):
        lineas.append(f"Documento: {datos['objeto_id_confirmado']}")
    if "intentos_redactor" in datos:
        lineas.append(f"Intento del redactor: {datos['intentos_redactor']}")
    if "vueltas_modificacion" in datos:
        lineas.append(f"Vuelta de modificación: {datos['vueltas_modificacion']}")
    contenido = datos.get("contenido_adaptado")
    if isinstance(contenido, dict) and contenido:
        lineas.append(
            f"Contenido: «{contenido.get('titulo', 'sin título')}» "
            f"({len(contenido.get('items') or [])} ítems)"
        )
    meta = datos.get("metadatos")
    if isinstance(meta, dict) and meta.get("conceptos_clave"):
        lineas.append(f"{len(meta['conceptos_clave'])} conceptos clave")
    alm = datos.get("almacenamiento_oci")
    if isinstance(alm, dict) and alm.get("objeto_id"):
        lineas.append(f"Guardado en OCI: {alm['objeto_id']}")
    if "imagenes_items" in datos:
        lineas.append(f"{len(datos['imagenes_items'] or {})} ícono(s) encontrado(s)")
    if datos.get("status"):
        lineas.append(f"Status: {datos['status']}")
    if datos.get("error"):
        lineas.append(f"⚠ Error: {datos['error']}")
    chunks = datos.get("chunks_fuente_confirmados")
    if isinstance(chunks, list):
        lineas.append(f"{len(chunks)} chunk(s) recuperado(s)")
        for c in chunks[:5]:
            if isinstance(c, dict):
                texto = " ".join(str(c.get("texto", "")).split())[:90]
                lineas.append(f"· {c.get('id', '?')}: {texto}…")
    if datos.get("perfil_destinatario"):
        lineas.append(f"Perfil: {datos['perfil_destinatario']}")
    if datos.get("formato_salida"):
        lineas.append(f"Formato: {datos['formato_salida']}")
    if datos.get("nicho_sector"):
        lineas.append(f"Nicho: {datos['nicho_sector']}")
    if datos.get("nivel_detalle"):
        lineas.append(f"Nivel de detalle: {datos['nivel_detalle']}")
    if datos.get("tema_consulta"):
        lineas.append(f"Tema: {datos['tema_consulta']}")
    if not lineas:
        lineas.append("Actualizó: " + ", ".join(datos.keys()))
    return lineas


def _html_nodo(item: dict, activo: bool = False) -> str:
    nodo = item.get("nodo", "")
    es_error = nodo == "_error" or any(
        str(d).startswith("⚠ Error") for d in (item.get("detalle") or [])
    )
    if es_error:
        clase = "error"
        icono = "⚠"
    elif activo:
        clase = "activo"
        icono = "⚡"
    else:
        clase = ""
        icono = "⚡"
    nombre = NOMBRES_NODOS.get(nodo, nodo)
    detalle = "<br>".join(html.escape(str(linea)) for linea in item.get("detalle") or [])
    return f"""
    <div class="nm-nodo {clase}">
        <div class="nm-nodo-titulo">{icono} {html.escape(nombre)}</div>
        <div class="nm-nodo-detalle">{detalle}</div>
    </div>
    """


def _html_nodo_en_curso(nombre_nodo: str) -> str:
    return f"""
    <div class="nm-nodo en-curso">
        <div class="nm-nodo-titulo">⏳ {html.escape(NOMBRES_NODOS.get(nombre_nodo, nombre_nodo))}</div>
        <div class="nm-nodo-detalle">en curso…</div>
    </div>
    """


def _pintar_nodo(contenedor, nombre_nodo: str, activo: bool, datos=None, slot=None):
    if nombre_nodo not in NOMBRES_NODOS:
        return
    item = {"nodo": nombre_nodo, "detalle": _resumen_nodo(datos)}
    timeline = st.session_state.setdefault("timeline_nodos", [])
    timeline.append(item)
    # Tope para no saturar la UI en sesiones largas
    if len(timeline) > MAX_TIMELINE_NODOS:
        st.session_state.timeline_nodos = timeline[-MAX_TIMELINE_NODOS:]
    html_item = _html_nodo(item, activo)
    if slot is not None:
        slot.markdown(html_item, unsafe_allow_html=True)
    else:
        with contenedor:
            st.markdown(html_item, unsafe_allow_html=True)


def correr(coro):
    """Ejecuta una corrutina desde el hilo síncrono de Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Evita conflicto si ya hay un loop (p. ej. entornos embebidos)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Helpers de UI
# --------------------------------------------------------------------------

_RESPUESTAS_VACIAS = {
    "", "no", "nada", "ninguno", "ninguna", "listo", "ok", "okey", "fin",
    "termine", "terminar", "no gracias", "gracias", "esta bien", "ya esta",
}


def _es_vacio(texto: str | None) -> bool:
    normalizado = re.sub(r"[.,;:!¡?¿]", "", (texto or "").lower()).strip()
    normalizado = normalizado.replace("á", "a").replace("é", "e").replace("í", "i")
    normalizado = normalizado.replace("ó", "o").replace("ú", "u")
    return normalizado in _RESPUESTAS_VACIAS


def _fase_desde_interrupt(valor: Any) -> str:
    """
    Mapea el valor de interrupt del grafo a una fase de UI.

    Contrato preferido: el grafo siempre envía {"tipo": "<uno de TIPOS_INTERRUPT>", ...}.
    Se mantiene fallback por claves legacy para no romper versiones anteriores del grafo.
    """
    if not isinstance(valor, dict):
        return "esperando_ejecucion"

    tipo = valor.get("tipo")
    if tipo in TIPOS_INTERRUPT:
        return {
            "pedir_parametro": "esperando_parametro",
            "confirmar_ejecucion": "esperando_ejecucion",
            "confirmar_aclaracion": "esperando_aclaracion",
            "confirmar_modificacion": "esperando_modificacion",
        }[tipo]

    # Fallbacks legacy (sin "tipo")
    if "mensaje_aclaracion" in valor:
        return "esperando_aclaracion"
    if "vueltas_modificacion" in valor:
        return "esperando_modificacion"
    if "parametro" in valor or "opciones" in valor:
        return "esperando_parametro"
    if "objeto_id_confirmado" in valor or "parametros" in valor:
        return "esperando_ejecucion"

    logger.warning("Interrupt sin tipo reconocido: keys=%s", list(valor.keys()))
    return "esperando_ejecucion"


def _aplicar_estado(estado) -> None:
    st.session_state.estado_final = estado
    values = (estado.values or {}) if estado is not None else {}
    st.session_state.imagenes_items = {
        int(k): v
        for k, v in (values.get("imagenes_items") or {}).items()
    }
    tasks = getattr(estado, "tasks", None) or []
    if tasks and getattr(tasks[0], "interrupts", None):
        valor = tasks[0].interrupts[0].value
        st.session_state.interrupt_actual = valor
        st.session_state.fase = _fase_desde_interrupt(valor)
    else:
        st.session_state.interrupt_actual = None
        st.session_state.fase = "fin"


def _validar_upload(archivo) -> str | None:
    """Devuelve mensaje de error o None si el archivo es aceptable."""
    if archivo is None:
        return "No se recibió ningún archivo."
    nombre = (archivo.name or "").lower()
    if not any(nombre.endswith(ext) for ext in EXTENSIONES_PERMITIDAS):
        return f"Extensión no permitida. Usa: {', '.join(sorted(EXTENSIONES_PERMITIDAS))}"
    # Streamlit UploadedFile expone size en bytes cuando está disponible
    size = getattr(archivo, "size", None)
    if size is not None and size > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return f"El archivo supera el límite de {mb:.0f} MB."
    return None


def _reset_sesion() -> None:
    st.session_state.fase = "inicio"
    st.session_state.interrupt_actual = None
    st.session_state.estado_final = None
    st.session_state.timeline_nodos = []
    st.session_state.imagenes_items = {}
    st.session_state.thread_id = "sesion-" + uuid.uuid4().hex


# ==========================================================================
# BLOQUE DE RENDERIZADO
# ==========================================================================

def _tipo_item(item: dict) -> str:
    if "paso_numero" in item:
        return "tutorial"
    if "frente" in item:
        return "flashcard"
    if "pregunta" in item:
        return "quiz"
    if "punto_clave" in item:
        return "resumen"
    return "generico"


def _renderizar_item(indice: int, item: dict, imagen=None):
    tipo = _tipo_item(item)
    if tipo == "tutorial":
        st.markdown(f"**Paso {item.get('paso_numero', indice)}. {item.get('titulo_paso', '')}**")
        st.write(item.get("instruccion", ""))
        if item.get("codigo_ejemplo"):
            st.code(item["codigo_ejemplo"])
        if item.get("resultado_esperado"):
            st.caption(f"✔ Resultado esperado: {item['resultado_esperado']}")
        if item.get("advertencia"):
            st.warning(item["advertencia"])
    elif tipo == "flashcard":
        # La columna del ícono se mantiene siempre: todas las cards quedan alineadas
        # aunque alguna no tenga ícono.
        col_img, col_card = st.columns([1, 8])
        if imagen:
            col_img.image(imagen, width=48)
        with col_card:
            with st.expander(f"🃏 {item.get('frente', '')}"):
                st.write(item.get("dorso", ""))
                if item.get("pista_didactica"):
                    st.caption(f"💡 {item['pista_didactica']}")
    elif tipo == "quiz":
        st.markdown(f"**{indice}. {item.get('pregunta', '')}**")
        for opcion in item.get("opciones") or []:
            if isinstance(opcion, dict):
                st.markdown(f"- **{opcion.get('id', '')})** {opcion.get('texto', '')}")
        with st.expander("Ver respuesta y justificación"):
            st.markdown(f"**Respuesta correcta:** {item.get('respuesta_correcta', '')}")
            st.write(item.get("justificacion", ""))
            if item.get("analisis_distractores"):
                st.caption(f"Por qué las otras no: {item['analisis_distractores']}")
    elif tipo == "resumen":
        st.markdown(f"**{item.get('punto_clave', '')}**")
        if item.get("implicacion"):
            st.write(f"➜ {item['implicacion']}")
        if item.get("relevancia_negocio"):
            st.success(f"💼 {item['relevancia_negocio']}")
        if item.get("recomendacion"):
            st.info(f"→ {item['recomendacion']}")
    else:
        st.json(item)
    anclaje = item.get("anclaje")
    if anclaje:
        with st.expander("Ver fuente"):
            st.caption("Respaldo en la fuente: " + ", ".join(str(a) for a in anclaje))


def _salida_estructurada(values: dict) -> dict:
    return {
        "status": values.get("status"),
        "metadatos": values.get("metadatos") or {},
        "contenido_adaptado": values.get("contenido_adaptado") or {},
        "evaluacion_calidad": values.get("evaluacion_calidad") or {},
        "almacenamiento_oci": values.get("almacenamiento_oci") or {},
    }


def _mostrar_json_crudo(values: dict):
    salida = _salida_estructurada(values)
    texto = json.dumps(salida, ensure_ascii=False, indent=2, default=str)
    with st.expander("🛠 JSON técnico (depuración)", expanded=False):
        tab_texto, tab_arbol = st.tabs(["Texto", "Árbol"])
        with tab_texto:
            st.code(texto, language="json")
        with tab_arbol:
            st.json(salida, expanded=True)
        st.download_button(
            "⬇ Descargar JSON",
            data=texto,
            file_name="salida_estructurada.json",
            mime="application/json",
        )


def _html_ficha_reporte(
    resultado: dict,
    metadatos: dict,
    evaluacion: dict | None = None,
    almacenamiento: dict | None = None,
    status: str = "exito",
) -> str:
    """Ficha/reporte ejecutivo con BLUF, hallazgos, recomendaciones y evaluación."""
    def esc(v) -> str:
        return html.escape(str(v)) if v is not None else ""

    items = [i for i in (resultado.get("items") or []) if isinstance(i, dict)]
    evaluacion = evaluacion or {}
    almacenamiento = almacenamiento or {}

    chips_meta = []
    if metadatos.get("formato_generado"):
        chips_meta.append(f'<span class="nm-chip">{esc(metadatos["formato_generado"])}</span>')
    if metadatos.get("perfil_aplicado"):
        chips_meta.append(f'<span class="nm-chip">Perfil: {esc(metadatos["perfil_aplicado"])}</span>')
    if metadatos.get("nivel_detalle_aplicado"):
        chips_meta.append(f'<span class="nm-chip">Nivel: {esc(metadatos["nivel_detalle_aplicado"])}</span>')
    if metadatos.get("nicho_aplicado"):
        chips_meta.append(f'<span class="nm-chip">Nicho: {esc(metadatos["nicho_aplicado"])}</span>')
    tiempo = metadatos.get("tiempo_estimado_estudio_minutos")
    if tiempo:
        chips_meta.append(f'<span class="nm-chip">⏱ {esc(tiempo)} min</span>')
    if metadatos.get("idioma"):
        chips_meta.append(f'<span class="nm-chip">Idioma: {esc(metadatos["idioma"])}</span>')

    # Hallazgos
    hallazgos_html = []
    fuentes = []
    for n, item in enumerate(items, start=1):
        detalle = []
        if item.get("implicacion"):
            detalle.append(
                f'<div class="nm-hallazgo-fila"><b>Implicación</b>{esc(item["implicacion"])}</div>'
            )
        if item.get("relevancia_negocio"):
            detalle.append(
                f'<div class="nm-hallazgo-fila"><b>Impacto negocio</b>{esc(item["relevancia_negocio"])}</div>'
            )
        if item.get("recomendacion"):
            detalle.append(
                f'<div class="nm-hallazgo-fila"><b>Recomendación</b>{esc(item["recomendacion"])}</div>'
            )
        anclaje = [str(a) for a in (item.get("anclaje") or [])]
        fuentes.extend(anclaje)
        if anclaje:
            detalle.append(
                f'<div class="nm-hallazgo-fuente">Fuente: {esc(", ".join(anclaje))}</div>'
            )
        hallazgos_html.append(
            '<div class="nm-hallazgo">'
            f'<div class="nm-hallazgo-num">{n}</div>'
            '<div>'
            f'<div class="nm-hallazgo-titulo">{esc(item.get("punto_clave", ""))}</div>'
            f'{"".join(detalle)}'
            '</div></div>'
        )

    conceptos = metadatos.get("conceptos_clave") or []
    chips_conceptos = "".join(f'<span class="nm-chip">{esc(c)}</span>' for c in conceptos)

    prerreq = metadatos.get("prerrequisitos") or []
    prerreq_txt = " · ".join(esc(p) for p in prerreq) if prerreq else ""

    # Recomendaciones prioritarias (lista global)
    recos = resultado.get("recomendaciones_prioritarias") or []
    recos_html = ""
    if recos:
        bloques = "".join(f'<div class="nm-reco">{esc(r)}</div>' for r in recos if r)
        recos_html = f'<div class="nm-rep-seccion">Recomendaciones prioritarias</div>{bloques}'

    # Evaluación
    eval_html = ""
    if evaluacion:
        score = evaluacion.get("anclaje_fuente_score")
        claridad = evaluacion.get("claridad_pedagogica")
        soportadas = evaluacion.get("afirmaciones_soportadas")
        evaluadas = evaluacion.get("afirmaciones_evaluadas")
        supero = evaluacion.get("supero_umbral")
        reintentos = evaluacion.get("reintentos_realizados")
        obs = evaluacion.get("observaciones")
        bluf = evaluacion.get("bluf_presente")
        idioma_ok = evaluacion.get("idioma_consistente")

        badge_umbral = (
            '<span class="nm-badge completado">Umbral superado</span>'
            if supero else
            '<span class="nm-badge pendiente">Umbral no superado</span>'
        )
        metricas_eval = []
        if score is not None:
            metricas_eval.append(f"Anclaje fuente: <b>{esc(score)}</b>")
        if claridad:
            metricas_eval.append(f"Claridad: <b>{esc(claridad)}</b>")
        if evaluadas is not None and soportadas is not None:
            metricas_eval.append(f"Afirmaciones: <b>{esc(soportadas)}/{esc(evaluadas)}</b>")
        if reintentos is not None:
            metricas_eval.append(f"Reintentos: <b>{esc(reintentos)}</b>")
        if bluf is not None:
            metricas_eval.append(f"BLUF: <b>{'Sí' if bluf else 'No'}</b>")
        if idioma_ok is not None:
            metricas_eval.append(f"Idioma consistente: <b>{'Sí' if idioma_ok else 'No'}</b>")

        eval_html = (
            '<div class="nm-rep-seccion">Evaluación de calidad</div>'
            f'<div style="margin-bottom:12px">{badge_umbral}</div>'
            f'<div class="nm-rep-resumen">{" · ".join(metricas_eval)}</div>'
        )
        if obs:
            eval_html += (
                f'<div class="nm-hallazgo-fila" style="margin-top:10px">'
                f'<b>Observaciones</b>{esc(obs)}</div>'
            )

    alm_html = ""
    if almacenamiento.get("objeto_id"):
        alm_html = (
            '<div class="nm-rep-pie">'
            f'Guardado en OCI · bucket <code>{esc(almacenamiento.get("bucket", ""))}</code> · '
            f'<code>{esc(almacenamiento["objeto_id"])}</code>'
            f' · status: {esc(almacenamiento.get("status_upload", ""))}'
            '</div>'
        )
    elif fuentes:
        alm_html = (
            f'<div class="nm-rep-pie">Basado en {len(set(fuentes))} fragmento(s) del documento fuente.</div>'
        )

    status_badge = (
        '<span class="nm-badge completado">✅ Éxito</span>'
        if status == "exito" else
        f'<span class="nm-badge pendiente">{esc(status)}</span>'
    )

    # BLUF
    bluf_txt = resultado.get("mensaje_principal") or ""
    bluf_html = (
        f'<div class="nm-rep-seccion">Mensaje clave</div>'
        f'<div class="nm-bluf">{esc(bluf_txt)}</div>'
        if bluf_txt else ""
    )

    partes = [
        '<div class="nm-rep">',
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
        f'<div class="nm-rep-kicker">Reporte ejecutivo</div>{status_badge}</div>',
        f'<h2 class="nm-rep-titulo">{esc(resultado.get("titulo", "Sin título"))}</h2>',
        f'<div class="nm-rep-meta">{"".join(chips_meta)}</div>',
    ]

    if bluf_html:
        partes.append(bluf_html)

    if resultado.get("introduccion_contextualizada"):
        partes += [
            '<div class="nm-rep-seccion">Resumen</div>',
            f'<div class="nm-rep-resumen">{esc(resultado["introduccion_contextualizada"])}</div>',
        ]

    if hallazgos_html:
        partes += ['<div class="nm-rep-seccion">Hallazgos clave</div>', *hallazgos_html]

    if recos_html:
        partes.append(recos_html)

    if chips_conceptos:
        partes += [
            '<div class="nm-rep-seccion">Conceptos clave</div>',
            f'<div>{chips_conceptos}</div>',
        ]

    if prerreq_txt:
        partes += [
            '<div class="nm-rep-seccion">Prerrequisitos</div>',
            f'<div class="nm-rep-resumen">{prerreq_txt}</div>',
        ]

    if eval_html:
        partes.append(eval_html)

    if alm_html:
        partes.append(alm_html)

    partes.append('</div>')
    return "".join(partes)


@st.cache_data(show_spinner=False, ttl=3600)
def _svg_inline(url: str) -> str:
    """
    Descarga el SVG y lo devuelve como data URI, para que el HTML descargado sea
    autónomo (se ve sin internet y al imprimir). Si falla, devuelve la URL original.
    """
    try:
        import httpx

        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
        return "data:image/svg+xml;base64," + base64.b64encode(r.content).decode("ascii")
    except Exception:
        return url


def _html_item_imprimible(n: int, item: dict, imagen: str | None, esc) -> str:
    """Un ítem del paquete como HTML imprimible, según su tipo."""
    tipo = _tipo_item(item)
    if tipo == "flashcard":
        icono = f'<img class="icono" src="{esc(imagen)}" alt="">' if imagen else ""
        pista = (
            f'<p class="pista">💡 {esc(item["pista_didactica"])}</p>'
            if item.get("pista_didactica") else ""
        )
        cuerpo = (
            f'<div class="card">{icono}<div>'
            f'<h3>{n}. {esc(item.get("frente", ""))}</h3>'
            f'<p>{esc(item.get("dorso", ""))}</p>{pista}</div></div>'
        )
    elif tipo == "quiz":
        opciones = "".join(
            f"<li><strong>{esc(o.get('id', ''))})</strong> {esc(o.get('texto', ''))}</li>"
            for o in (item.get("opciones") or []) if isinstance(o, dict)
        )
        cuerpo = (
            f"<h3>{n}. {esc(item.get('pregunta', ''))}</h3><ul>{opciones}</ul>"
            f"<p><strong>Respuesta correcta:</strong> {esc(item.get('respuesta_correcta', ''))}</p>"
            f"<p>{esc(item.get('justificacion', ''))}</p>"
        )
        if item.get("analisis_distractores"):
            cuerpo += f"<p class='pista'>Por qué las otras no: {esc(item['analisis_distractores'])}</p>"
    elif tipo == "tutorial":
        cuerpo = (
            f"<h3>Paso {esc(item.get('paso_numero', n))}. {esc(item.get('titulo_paso', ''))}</h3>"
            f"<p>{esc(item.get('instruccion', ''))}</p>"
        )
        if item.get("resultado_esperado"):
            cuerpo += f"<p><strong>Resultado esperado:</strong> {esc(item['resultado_esperado'])}</p>"
        if item.get("advertencia"):
            cuerpo += f"<p><strong>Advertencia:</strong> {esc(item['advertencia'])}</p>"
    elif tipo == "resumen":
        cuerpo = f"<h3>{n}. {esc(item.get('punto_clave', ''))}</h3>"
        if item.get("implicacion"):
            cuerpo += f"<p><strong>Implicación:</strong> {esc(item['implicacion'])}</p>"
        if item.get("relevancia_negocio"):
            cuerpo += f"<p><strong>Impacto en el negocio:</strong> {esc(item['relevancia_negocio'])}</p>"
        if item.get("recomendacion"):
            cuerpo += f"<p><strong>Recomendación:</strong> {esc(item['recomendacion'])}</p>"
    else:
        cuerpo = f"<pre>{esc(json.dumps(item, ensure_ascii=False, indent=2, default=str))}</pre>"

    anclaje = item.get("anclaje") or []
    if anclaje:
        cuerpo += (
            f"<p style='color:#666;font-size:0.85em'>Fuente: "
            f"{esc(', '.join(str(a) for a in anclaje))}</p>"
        )
    return cuerpo


def _html_imprimible(
    resultado: dict,
    metadatos: dict,
    evaluacion: dict | None = None,
    almacenamiento: dict | None = None,
    status: str = "exito",
) -> str:
    def esc(v) -> str:
        return html.escape(str(v)) if v is not None else ""

    items = [i for i in (resultado.get("items") or []) if isinstance(i, dict)]
    evaluacion = evaluacion or {}
    almacenamiento = almacenamiento or {}

    meta_lineas = []
    if metadatos.get("formato_generado"):
        meta_lineas.append(f"<strong>Formato:</strong> {esc(metadatos['formato_generado'])}")
    if metadatos.get("perfil_aplicado"):
        meta_lineas.append(f"<strong>Perfil:</strong> {esc(metadatos['perfil_aplicado'])}")
    if metadatos.get("nivel_detalle_aplicado"):
        meta_lineas.append(f"<strong>Nivel:</strong> {esc(metadatos['nivel_detalle_aplicado'])}")
    if metadatos.get("nicho_aplicado"):
        meta_lineas.append(f"<strong>Nicho:</strong> {esc(metadatos['nicho_aplicado'])}")
    if metadatos.get("tiempo_estimado_estudio_minutos"):
        meta_lineas.append(
            f"<strong>Tiempo estimado:</strong> {esc(metadatos['tiempo_estimado_estudio_minutos'])} min"
        )
    if metadatos.get("idioma"):
        meta_lineas.append(f"<strong>Idioma:</strong> {esc(metadatos['idioma'])}")

    imagenes = st.session_state.get("imagenes_items") or {}
    hallazgos = []
    for n, item in enumerate(resultado.get("items") or [], start=1):
        if not isinstance(item, dict):
            continue
        imagen = imagenes.get(n)
        if imagen:
            # color más oscuro para que se vea sobre fondo blanco al imprimir
            imagen = _svg_inline(imagen.replace("%234FC3F7", "%231A56DB"))
        hallazgos.append(_html_item_imprimible(n, item, imagen, esc))

    tipos = {_tipo_item(i) for i in items}
    titulo_seccion = {
        "flashcard": "Tarjetas de estudio",
        "tutorial": "Pasos",
        "quiz": "Preguntas",
    }.get(next(iter(tipos)) if len(tipos) == 1 else "", "Hallazgos clave")

    conceptos = metadatos.get("conceptos_clave") or []
    prerreq = metadatos.get("prerrequisitos") or []
    recos = resultado.get("recomendaciones_prioritarias") or []
    bluf = resultado.get("mensaje_principal") or ""

    eval_bloque = ""
    if evaluacion:
        eval_bloque = f"""
        <h2>Evaluación de calidad</h2>
        <ul>
            <li>Anclaje fuente: <strong>{esc(evaluacion.get('anclaje_fuente_score'))}</strong></li>
            <li>Claridad pedagógica: <strong>{esc(evaluacion.get('claridad_pedagogica'))}</strong></li>
            <li>Afirmaciones soportadas: <strong>{esc(evaluacion.get('afirmaciones_soportadas'))}/{esc(evaluacion.get('afirmaciones_evaluadas'))}</strong></li>
            <li>Reintentos: <strong>{esc(evaluacion.get('reintentos_realizados'))}</strong></li>
            <li>Umbral superado: <strong>{"Sí" if evaluacion.get("supero_umbral") else "No"}</strong></li>
        </ul>
        """
        if evaluacion.get("observaciones"):
            eval_bloque += f"<p><strong>Observaciones:</strong> {esc(evaluacion['observaciones'])}</p>"

    alm_bloque = ""
    if almacenamiento.get("objeto_id"):
        alm_bloque = f"""
        <p style="margin-top:32px;font-size:0.85em;color:#555">
            Guardado en OCI · bucket <code>{esc(almacenamiento.get('bucket'))}</code> ·
            <code>{esc(almacenamiento['objeto_id'])}</code>
        </p>
        """

    bluf_bloque = f"<div class='intro'><strong>Mensaje clave:</strong> {esc(bluf)}</div>" if bluf else ""
    recos_bloque = ""
    if recos:
        recos_bloque = "<h2>Recomendaciones prioritarias</h2><ul>" + "".join(
            f"<li>{esc(r)}</li>" for r in recos if r
        ) + "</ul>"

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{esc(resultado.get("titulo", "Reporte NuevaMente"))}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.15rem; text-transform: uppercase; letter-spacing: 0.06em; color: #555; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
  h3 {{ font-size: 1.05rem; margin-top: 22px; }}
  .meta {{ color: #555; font-size: 0.95rem; margin-bottom: 24px; }}
  .meta span {{ margin-right: 18px; }}
  .intro {{ background: #f5f7fa; padding: 16px 20px; border-radius: 8px; margin: 20px 0; }}
  .chip {{ display: inline-block; background: #e8f0fe; color: #1a56db; border-radius: 999px; padding: 3px 12px; font-size: 0.85rem; margin: 2px 4px 2px 0; }}
  .card {{ display: flex; gap: 14px; align-items: flex-start; border: 1px solid #ddd; border-radius: 10px; padding: 12px 16px; margin: 12px 0; break-inside: avoid; }}
  .card h3 {{ margin: 0 0 6px 0; }}
  .card p {{ margin: 4px 0; }}
  .card .icono {{ width: 40px; height: 40px; flex: 0 0 40px; }}
  .pista {{ color: #555; font-size: 0.92em; }}
  @media print {{
    body {{ margin: 0; padding: 12px; }}
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>
  <h1>{esc(resultado.get("titulo", "Sin título"))}</h1>
  <div class="meta">{"".join(f"<span>{m}</span>" for m in meta_lineas)}</div>
  {bluf_bloque}
  {"<div class='intro'>" + esc(resultado.get("introduccion_contextualizada", "")) + "</div>" if resultado.get("introduccion_contextualizada") else ""}
  <h2>{esc(titulo_seccion)}</h2>
  {"".join(hallazgos)}
  {recos_bloque}
  {"<h2>Conceptos clave</h2><div>" + "".join(f"<span class='chip'>{esc(c)}</span>" for c in conceptos) + "</div>" if conceptos else ""}
  {"<h2>Prerrequisitos</h2><p>" + " · ".join(esc(p) for p in prerreq) + "</p>" if prerreq else ""}
  {eval_bloque}
  {alm_bloque}
  <p class="no-print" style="margin-top:40px;font-size:0.8em;color:#888">
    Generado por NuevaMente · {esc(status)}
  </p>
</body>
</html>"""


def _renderizar_contenido(
    resultado: dict,
    metadatos: dict,
    evaluacion: dict | None = None,
    almacenamiento: dict | None = None,
    status: str = "exito",
):
    items_dict = [i for i in (resultado.get("items") or []) if isinstance(i, dict)]

    if tipo_documento(items_dict):
        st.markdown(
            html_documento(resultado, metadatos, items_dict, evaluacion, almacenamiento, status),
            unsafe_allow_html=True,
        )
        return

    if items_dict and all(_tipo_item(i) == "resumen" for i in items_dict):
        st.markdown(
            _html_ficha_reporte(
                resultado, metadatos,
                evaluacion=evaluacion,
                almacenamiento=almacenamiento,
                status=status,
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"## {resultado.get('titulo', 'Sin título')}")
    if metadatos.get("formato_generado"):
        st.caption(f"Formato: {metadatos['formato_generado']}")

    tiempo = metadatos.get("tiempo_estimado_estudio_minutos")
    metricas = [
        ("⏱ Estudio", f"{tiempo} min" if tiempo else None),
        ("Perfil", metadatos.get("perfil_aplicado")),
        ("Nicho", metadatos.get("nicho_aplicado")),
        ("Nivel de detalle", metadatos.get("nivel_detalle_aplicado")),
    ]
    metricas = [(e, v) for e, v in metricas if v]
    if metricas:
        for col, (etiqueta, valor) in zip(st.columns(len(metricas)), metricas):
            col.metric(etiqueta, valor)

    conceptos = metadatos.get("conceptos_clave") or []
    if conceptos:
        chips = "".join(f'<span class="nm-chip">{html.escape(str(c))}</span>' for c in conceptos)
        st.markdown(chips, unsafe_allow_html=True)

    prerrequisitos = metadatos.get("prerrequisitos") or []
    if prerrequisitos:
        st.info("**Prerrequisitos:** " + " · ".join(str(p) for p in prerrequisitos))

    if resultado.get("mensaje_principal"):
        st.success(f"**Mensaje clave:** {resultado['mensaje_principal']}")

    introduccion = resultado.get("introduccion_contextualizada")
    if introduccion:
        with st.container(border=True):
            st.write(introduccion)

    imagenes = st.session_state.get("imagenes_items") or {}
    for indice, item in enumerate(resultado.get("items") or [], start=1):
        if isinstance(item, dict):
            with st.container(border=True):
                _renderizar_item(indice, item, imagen=imagenes.get(indice))

    recos = resultado.get("recomendaciones_prioritarias") or []
    if recos:
        st.markdown("#### Recomendaciones prioritarias")
        for r in recos:
            st.markdown(f"- {r}")


def _botones_accion_reporte(
    resultado: dict,
    metadatos: dict,
    evaluacion: dict | None = None,
    almacenamiento: dict | None = None,
    status: str = "exito",
):
    html_print = _html_imprimible(
        resultado, metadatos, evaluacion, almacenamiento, status
    )
    titulo_safe = re.sub(r"[^\w\-]+", "_", (resultado.get("titulo") or "reporte")[:60])
    nombre_archivo = f"{titulo_safe}.html"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="🖨 Imprimir / Descargar",
            data=html_print,
            file_name=nombre_archivo,
            mime="text/html",
            use_container_width=True,
            help="Descarga el reporte en HTML limpio. Ábrelo y usa Ctrl+P / Cmd+P para imprimir o Guardar como PDF.",
        )

    with col2:
        if st.button("☁ Guardar en OCI", use_container_width=True, type="secondary"):
            with st.spinner("Subiendo reporte a Object Storage…"):
                try:
                    res = correr(_subir_reporte_generado(nombre_archivo, html_print, USUARIO.user_id))
                    objeto = res.get("objeto_id") or res.get("path") or str(res)
                    st.success(f"✅ Guardado en OCI: `{objeto}`")
                except Exception as e:
                    st.error(f"Error al subir: {e}")

    with col3:
        if st.button("🔄 Otra consulta", use_container_width=True):
            _reset_sesion()
            st.rerun()


def _mostrar_resultado(
    values: dict,
    *,
    permitir_modificar: bool = False,
    interrupt_data: dict | None = None,
) -> None:
    """Render unificado de resultado (fase fin y esperando_modificacion)."""
    resultado = values.get("contenido_adaptado") or {}
    metadatos = values.get("metadatos") or {}
    evaluacion = values.get("evaluacion_calidad") or {}
    almacenamiento = values.get("almacenamiento_oci") or {}
    status = values.get("status") or ("exito" if permitir_modificar else "fin")
    error = values.get("error")
    interrupt_data = interrupt_data or {}

    if error:
        st.markdown(
            '<span class="nm-badge pendiente">⚠ Error</span>',
            unsafe_allow_html=True,
        )
    elif status == "cancelado" and not resultado:
        st.markdown(
            '<span class="nm-badge pendiente">Pedido cancelado</span>',
            unsafe_allow_html=True,
        )
    elif permitir_modificar:
        st.markdown(
            '<span class="nm-badge completado">Contenido generado</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="nm-badge completado">✅ Listo</span>',
            unsafe_allow_html=True,
        )

    if error:
        st.warning(f"El flujo terminó con error: {error}")
    elif not resultado:
        if status == "cancelado":
            st.info("Cancelaste el pedido. No se generó contenido.")
        else:
            st.info(f"Flujo terminado (status: `{status}`). No hay contenido para mostrar.")
    else:
        try:
            _renderizar_contenido(resultado, metadatos, evaluacion, almacenamiento, status)
            _botones_accion_reporte(resultado, metadatos, evaluacion, almacenamiento, status)
            _mostrar_json_crudo(values)
        except Exception as exc:
            logger.exception("Error al renderizar resultado")
            st.error("No se pudo mostrar el contenido generado.")
            st.caption(str(exc))

    if interrupt_data.get("aviso"):
        st.warning(interrupt_data["aviso"])

    if not permitir_modificar and (error or not resultado):
        if st.button("🔄 Nueva consulta", key="nueva_consulta_fin"):
            _reset_sesion()
            st.rerun()

    if almacenamiento.get("objeto_id"):
        st.caption(f"Guardado en: `{almacenamiento.get('objeto_id', '')}`")
    elif not error and resultado:
        st.caption(
            "Sin objeto guardado en OCI (el flujo no pasó por guardado final o falló la carga)."
        )


# --------------------------------------------------------------------------
# Estado de sesión
# --------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "sesion-" + uuid.uuid4().hex
if "fase" not in st.session_state:
    st.session_state.fase = "inicio"
if "interrupt_actual" not in st.session_state:
    st.session_state.interrupt_actual = None
if "estado_final" not in st.session_state:
    st.session_state.estado_final = None
if "timeline_nodos" not in st.session_state:
    st.session_state.timeline_nodos = []

# --------------------------------------------------------------------------
# UI - Navegación principal
# --------------------------------------------------------------------------

def _cards_html(cards, extra: str = "") -> str:
    partes = []
    for icono, titulo, texto, *resto in cards:
        pie = f'<div class="nm-card-pie">{resto[0]}</div>' if resto else ""
        partes.append(
            f'<div class="nm-card"><div class="nm-card-icono">{icono}</div>'
            f'<div class="nm-card-titulo">{titulo}</div>'
            f'<div class="nm-card-texto">{texto}</div>{pie}</div>'
        )
    return f'<div class="nm-grid {extra}">{"".join(partes)}</div>'


def _flow_html(pasos) -> str:
    partes = []
    for n, (titulo, texto, humano) in enumerate(pasos, start=1):
        marca = (
            '<span class="nm-step-humano">👤 tú decides</span>'
            if humano else '<span class="nm-step-auto">🤖 automático</span>'
        )
        clase = " humano" if humano else ""
        partes.append(
            f'<div class="nm-step{clase}"><div class="nm-step-num">{n}</div><div>'
            f'<div class="nm-step-titulo">{titulo}</div>'
            f'<div class="nm-step-texto">{texto}</div>{marca}</div></div>'
        )
    return f'<div class="nm-flow">{"".join(partes)}</div>'


# --------------------------------------------------------------------------
# Contenido estático de la pestaña Inicio (mosaico + trio)
# --------------------------------------------------------------------------

_MOSAICO_HTML = """<div class="nm-mosaic">
<div class="nm-tile t1"><svg viewBox="0 0 200 420" preserveAspectRatio="xMidYMid slice"><g stroke="#4FC3F7" stroke-opacity=".4" fill="none"><path d="M10 0L100 300M40 0L100 300M70 0L100 300M100 0L100 300M130 0L100 300M160 0L100 300M190 0L100 300"/></g><path d="M62 420L100 240L138 420" fill="none" stroke="#7FD8FF" stroke-width="12" stroke-linejoin="round"/></svg></div>
<div class="nm-tile t2"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><rect x="24" y="28" width="152" height="144" rx="10" fill="rgba(255,255,255,.05)" stroke="#4FC3F7" stroke-opacity=".5"/><g fill="#8B96A8" fill-opacity=".6"><rect x="40" y="48" width="90" height="7" rx="3"/><rect x="40" y="66" width="120" height="7" rx="3"/><rect x="40" y="84" width="100" height="7" rx="3"/></g><rect x="38" y="100" width="124" height="20" rx="5" fill="#4FC3F7" fill-opacity=".3" stroke="#7FD8FF"/><text x="118" y="114" font-size="10" fill="#7FD8FF" font-family="monospace">[c-12]</text><g fill="#8B96A8" fill-opacity=".6"><rect x="40" y="132" width="110" height="7" rx="3"/><rect x="40" y="148" width="70" height="7" rx="3"/></g></svg></div>
<div class="nm-tile t3"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><g stroke="#4FC3F7" stroke-opacity=".6" fill="none"><path d="M100 100L40 50M100 100L165 55M100 100L50 155M100 100L160 150M40 50L165 55M50 155L160 150"/></g><g fill="#7FD8FF"><circle cx="100" cy="100" r="13"/><circle cx="40" cy="50" r="8"/><circle cx="165" cy="55" r="8"/><circle cx="50" cy="155" r="8"/><circle cx="160" cy="150" r="8"/></g></svg></div>
<div class="nm-tile t4"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><g transform="rotate(-10 100 100)"><rect x="42" y="50" width="116" height="84" rx="10" fill="rgba(255,255,255,.04)" stroke="#4FC3F7" stroke-opacity=".4"/></g><g transform="rotate(6 100 100)"><rect x="42" y="56" width="116" height="84" rx="10" fill="#13243F" stroke="#7FD8FF"/><rect x="60" y="78" width="80" height="8" rx="4" fill="#7FD8FF"/><rect x="60" y="98" width="56" height="7" rx="3" fill="#8B96A8" fill-opacity=".7"/></g></svg></div>
<div class="nm-tile t5"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><g stroke="#4FC3F7" stroke-opacity=".5" fill="rgba(255,255,255,.04)"><rect x="28" y="40" width="144" height="30" rx="8"/><rect x="28" y="85" width="144" height="30" rx="8" fill="#4FC3F7" fill-opacity=".22" stroke="#7FD8FF" stroke-opacity="1"/><rect x="28" y="130" width="144" height="30" rx="8"/></g><g fill="none" stroke="#7FD8FF"><circle cx="48" cy="55" r="6"/><circle cx="48" cy="100" r="6" fill="#7FD8FF"/><circle cx="48" cy="145" r="6"/></g></svg></div>
<div class="nm-tile t6"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><g stroke="#4FC3F7" stroke-opacity=".6" stroke-width="3"><path d="M75 40v30M100 40v30M125 40v30M75 130v30M100 130v30M125 130v30M40 75h30M40 100h30M40 125h30M130 75h30M130 100h30M130 125h30"/></g><rect x="70" y="70" width="60" height="60" rx="8" fill="#0A121F" stroke="#7FD8FF" stroke-width="2"/><circle cx="100" cy="100" r="12" fill="#4FC3F7" fill-opacity=".5" stroke="#7FD8FF"/></svg></div>
<div class="nm-tile t7"><svg viewBox="0 0 200 200" preserveAspectRatio="xMidYMid slice"><g fill="#4FC3F7" fill-opacity=".55"><rect x="30" y="120" width="22" height="50" rx="4"/><rect x="62" y="98" width="22" height="72" rx="4"/><rect x="94" y="76" width="22" height="94" rx="4"/><rect x="126" y="52" width="22" height="118" rx="4" fill="#7FD8FF" fill-opacity=".9"/></g><path d="M30 105L62 86L94 62L126 38L160 28" fill="none" stroke="#7FD8FF" stroke-width="3" stroke-linecap="round"/></svg></div>
</div>"""

_HOME_HTML = """<div class="nm-home-head">
<h1 class="nm-home-title">Transform your documents <span class="acento">into learning materials</span></h1>
<p class="nm-home-sub">The same obstacles keep repeating: long documents, generic content, answers without sources, and manual reviews. An team of agents solves these for you.</p>
<span class="nm-pill">Start Learning Agentic AI →</span>
</div>""" + _MOSAICO_HTML + """<div class="nm-trio">
<div><div class="nm-trio-titulo">Los documentos largos frenan el estudio</div><div class="nm-trio-texto">Convertimos tu PDF en flashcards, quizzes, tutoriales o resúmenes ejecutivos listos para usar.</div><div class="nm-trio-pista">Mira la pestaña Casos de uso</div></div>
<div><div class="nm-trio-titulo">El contenido genérico no enseña</div><div class="nm-trio-texto">Redactamos para el perfil, nicho y nivel de detalle que eliges, y cada ítem queda anclado a tu fuente.</div><div class="nm-trio-pista">Mira la pestaña Agentes</div></div>
<div><div class="nm-trio-titulo">Revisar a mano no escala</div><div class="nm-trio-texto">Un Revisor mide anclaje y claridad y reintenta si hace falta. Tú apruebas o pides cambios.</div><div class="nm-trio-pista">Mira la pestaña Cómo funciona</div></div>
</div>"""


tab_inicio, tab_agentes, tab_como, tab_casos, tab_seguridad, tab_actividad = st.tabs([
    "Inicio",
    "Agentes",
    "Cómo funciona",
    "Casos de uso",
    "Seguridad",
    "Actividad Agentica",
])

# --------------------------------------------------------------------------
# Pestaña: Inicio
# --------------------------------------------------------------------------
with tab_inicio:
    st.markdown(_HOME_HTML, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña: Agentes
# --------------------------------------------------------------------------
with tab_agentes:
    st.markdown("### Un equipo, cada uno con un trabajo")
    st.write("Cada agente hace una sola cosa y le pasa el resultado al siguiente.")
    st.markdown(_cards_html([
        ("🔎", "Buscador de Documentos",
         "Encuentra en tu biblioteca el documento que mejor encaja con el tema.", "Entrada: tema"),
        ("🧬", "Investigador RAG",
         "Recupera los fragmentos relevantes y te pide aclaración si el tema no aparece.", "Salida: fragmentos"),
        ("✍️", "Redactor Pedagógico",
         "Escribe en el formato y para el perfil que elegiste, con cada ítem anclado a la fuente.", "Salida: contenido"),
        ("🧐", "Crítico / Revisor",
         "Mide anclaje y claridad. Si no llega al umbral, devuelve feedback y se reintenta.", "Salida: evaluación"),
        ("🎨", "Ilustrador",
         "Busca íconos para acompañar las flashcards.", "Opcional"),
        ("🛠", "Modificador",
         "Aplica los cambios que pides sobre el resultado y lo vuelve a revisar.", "Tú lo activas"),
    ]), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña: Cómo funciona
# --------------------------------------------------------------------------
with tab_como:
    st.markdown("### De un tema a material listo para estudiar")
    st.markdown(_flow_html([
        ("Cuéntanos qué quieres aprender", "Escribes un tema o subes tu propio documento.", True),
        ("Eliges parámetros", "Documento, perfil del destinatario y formato de salida, paso a paso.", True),
        ("Confirmas el pedido", "Revisas todo y ajustas nicho y nivel de detalle antes de ejecutar.", True),
        ("Se investiga la fuente", "El Investigador recupera los fragmentos relevantes del documento.", False),
        ("Se redacta", "El Redactor Pedagógico genera el contenido adaptado al perfil.", False),
        ("Se revisa", "El Revisor evalúa anclaje y claridad, y reintenta si hace falta.", False),
        ("Se guarda", "El resultado se guarda en Object Storage con su evaluación.", False),
        ("Ajustas lo que quieras", "Pides cambios y el Modificador los aplica.", True),
    ]), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña: Casos de uso (demo: mismo concepto, distinto formato)
# --------------------------------------------------------------------------
with tab_casos:
    st.markdown("### Un mismo concepto, cuatro formas de aprenderlo")
    st.caption("Ejemplo ilustrativo con el concepto «embeddings». Elige un formato:")
    formato_demo = st.radio(
        "Formato",
        ["Flashcards", "Quiz", "Tutorial", "Resumen Ejecutivo"],
        horizontal=True,
        label_visibility="collapsed",
        key="demo_formato",
    )
    with st.container(border=True):
        if formato_demo == "Flashcards":
            st.markdown("**🃏 ¿Qué es un embedding?**")
            st.write("Un vector numérico que representa el significado de un texto: textos parecidos quedan cerca entre sí.")
            st.caption("💡 Piénsalo como coordenadas en un mapa de significados.")
        elif formato_demo == "Quiz":
            st.markdown("**1. ¿Para qué sirve la similitud coseno en un RAG?**")
            st.markdown("- **a)** Comprimir documentos\n- **b)** Medir qué tan cerca están dos vectores\n- **c)** Traducir texto")
            with st.expander("Ver respuesta y justificación"):
                st.markdown("**Respuesta correcta:** b")
                st.write("Compara el vector de la pregunta con los de los fragmentos para traer los más parecidos.")
        elif formato_demo == "Tutorial":
            st.markdown("**Paso 1. Dividir el documento en fragmentos**")
            st.write("Corta el texto en partes de tamaño parecido, con un pequeño solapamiento entre ellas.")
            st.caption("✔ Resultado esperado: una lista de fragmentos con su ID.")
        else:
            st.markdown(
                '<div class="nm-bluf">Los embeddings permiten buscar por significado y no por palabras exactas.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="nm-reco">Indexa tus documentos una sola vez y reutiliza la búsqueda.</div>',
                        unsafe_allow_html=True)
    st.markdown(_cards_html([
        ("👩‍🏫", "Docentes", "Materiales listos a partir de sus propios documentos."),
        ("🏫", "Instituciones", "Contenido consistente y trazable hasta la fuente."),
        ("🏢", "Equipos de formación", "Resúmenes ejecutivos y guías para capacitar más rápido."),
    ]), unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña: Seguridad (unificada)
# --------------------------------------------------------------------------
with tab_seguridad:
    st.markdown("### Confianza por diseño")
    st.markdown(_cards_html([
        ("🧑‍💻", "Nada se ejecuta sin tu OK",
         "Antes de generar siempre te mostramos el pedido y esperas a confirmar."),
        ("📎", "Respaldo en la fuente",
         "Cada ítem apunta a los fragmentos del documento y el Revisor mide ese anclaje."),
        ("🙅", "Si no hay evidencia, no inventa",
         "Cuando la fuente no alcanza, el sistema se abstiene en vez de rellenar."),
        ("🚦", "Uso controlado",
         "Hay un límite de uso por sesión para evitar abusos."),
        ("🧾", "Errores sin detalles técnicos",
         "Ves un mensaje claro; el detalle técnico queda solo en el registro interno."),
    ]), unsafe_allow_html=True)
    st.markdown("### Privacidad y control")
    st.markdown("""
    Tus datos están protegidos.

    Los agentes solo usan la información que tú autorizas y nunca actúan sin tu control.

    Puedes revisar, pausar o eliminar cualquier acción en todo momento.
    """)

# --------------------------------------------------------------------------
# Pestaña: Actividad Agentica
# --------------------------------------------------------------------------
with tab_actividad:

    st.markdown("""
    <div class="nm-hero">
        <div class="nm-hero-content">
            <span class="nm-badge">🧠 NuevaMente</span>
            <h1 class="nm-hero-title">
                Contenido educativo
                <span class="linea-acento">generado con IA</span>
            </h1>
            <p class="nm-hero-sub">
                Escribe un tema, elige la fuente y deja que el pipeline de agentes
                redacte, revise y guarde el resultado por ti.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    en_resultado = st.session_state.fase in ("esperando_modificacion", "fin")

    if en_resultado:
        st.toggle("⛶ Ocultar actividad", key="pantalla_completa")
        # Actividad arriba, a todo el ancho; resultado abajo
        col_timeline = st.expander(
            "Actividad en vivo",
            expanded=not st.session_state.get("pantalla_completa"),
        )
        col_form = st.container()
    else:
        col_form, col_timeline = st.columns([1, 1])

    def _contenedor_timeline():
        with col_timeline:
            st.markdown("#### Actividad en vivo")
            return st.container()

    def _reanudar(payload: dict):
        """Reanuda el grafo con la respuesta del usuario, actualiza el estado y refresca la pantalla."""
        contenedor = _contenedor_timeline()
        try:
            estado = correr(_resumir_flujo(payload, st.session_state.thread_id, USUARIO.user_id, contenedor))
            _aplicar_estado(estado)
        except Exception as exc:
            logger.exception("Fallo al reanudar flujo")
            mensaje = getattr(exc, "mensaje_usuario", None)
            if mensaje:
                st.error(mensaje)
            else:
                st.error("Ocurrió un error al continuar el pipeline. Revisa la actividad en vivo.")
                st.caption(str(exc))
            st.session_state.fase = "fin"
            return
        st.rerun()

    with col_form:
        fase = st.session_state.fase

        if fase == "inicio":
            with st.expander("📎 ¿No encuentras el documento que buscas? Súbelo aquí"):
                archivo_subido = st.file_uploader(
                    "Elige un PDF, Markdown o texto",
                    type=["pdf", "md", "txt"],
                    help=f"Máximo {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
                if archivo_subido:
                    col_boton, col_nube = st.columns([1, 4], vertical_alignment="center")
                    subir = col_boton.button("Subir")
                    col_nube.caption("☁️ Se guardará en OCI Cloud")
                    if subir:
                        err = _validar_upload(archivo_subido)
                        if err:
                            st.error(err)
                        else:
                            with st.spinner(f"Subiendo {archivo_subido.name}..."):
                                try:
                                    data = archivo_subido.read()
                                    if len(data) > MAX_UPLOAD_BYTES:
                                        st.error(
                                            f"El archivo supera el límite de "
                                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                                        )
                                    else:
                                        resultado_subida = correr(
                                            _subir_documento(archivo_subido.name, data, USUARIO.user_id)
                                        )
                                        st.success(
                                            f"✅ Subido a `{resultado_subida['objeto_id']}` "
                                            "— ya puedes buscarlo por tema abajo."
                                        )
                                except Exception as e:
                                    logger.exception("Error al subir documento")
                                    st.error(f"No se pudo subir el archivo: {e}")

            tema = st.chat_input("¿Sobre qué tema quieres generar contenido?")
            if tema:
                tema_limpio = tema.strip()[:500]
                if not tema_limpio:
                    st.warning("Escribe un tema válido.")
                else:
                    st.session_state.timeline_nodos = []
                    contenedor = _contenedor_timeline()
                    try:
                        estado = correr(
                            _iniciar_flujo(
                                tema_limpio, st.session_state.thread_id, USUARIO.user_id, contenedor
                            )
                        )
                        _aplicar_estado(estado)
                        st.rerun()
                    except Exception as exc:
                        logger.exception("Fallo al iniciar flujo")
                        mensaje = getattr(exc, "mensaje_usuario", None)
                        if mensaje:
                            st.error(mensaje)
                        else:
                            st.error("No se pudo iniciar el pipeline. Revisa la actividad en vivo.")
                            st.caption(str(exc))

        elif fase == "esperando_parametro":
            datos_paso = st.session_state.interrupt_actual or {}
            parametro = datos_paso.get("parametro")
            opciones = datos_paso.get("opciones") or []

            st.markdown(
                f'<span class="nm-badge pendiente">Paso {datos_paso.get("paso", 1)} '
                f'de {datos_paso.get("total", 1)}</span>',
                unsafe_allow_html=True,
            )
            resueltos = datos_paso.get("resueltos") or {}
            if resueltos:
                chips = "".join(
                    f'<span class="nm-chip">{html.escape(str(k))}: {html.escape(str(v))}</span>'
                    for k, v in resueltos.items()
                )
                st.markdown(chips, unsafe_allow_html=True)

            with st.chat_message("assistant"):
                st.markdown(datos_paso.get("pregunta") or "Elige una opción")
            if datos_paso.get("aviso"):
                st.warning(datos_paso["aviso"])

            if parametro == "documento":
                valores = [o["valor"] for o in opciones]
                elegido = st.selectbox("Documentos encontrados", valores, key="param_documento")
                if st.button("Continuar ▶", type="primary", key="param_documento_continuar"):
                    _reanudar({"valor": elegido})
            else:
                columnas = st.columns(2)
                for i, op in enumerate(opciones):
                    with columnas[i % 2]:
                        if st.button(
                            op["etiqueta"],
                            key=f"param_{parametro}_{op['valor']}",
                            use_container_width=True,
                            help=op.get("ayuda") or None,
                        ):
                            _reanudar({"valor": op["valor"]})

            escrito = st.chat_input("O escribe tu respuesta aquí…")
            if escrito and escrito.strip():
                _reanudar({"valor": escrito.strip()})

            if st.button("Cancelar", key="param_cancelar"):
                _reanudar({"cancelar": True})

        elif fase == "esperando_ejecucion":
            datos_conf = st.session_state.interrupt_actual or {}
            objeto_id = datos_conf.get("objeto_id_confirmado")
            actuales = datos_conf.get("parametros") or {}
            opciones_conf = datos_conf.get("opciones") or {}

            st.markdown('<span class="nm-badge completado">Confirma tu pedido</span>', unsafe_allow_html=True)
            st.write(f"📄 `{objeto_id}`")
            st.caption("Perfil y formato los elegiste hace un momento. Nicho y nivel de detalle arrancan con valores por defecto: cambia cualquiera si quieres.")

            def _selector(titulo: str, campo: str):
                ops = opciones_conf.get(campo) or []
                valores = [o["valor"] for o in ops]
                etiquetas = {o["valor"]: o["etiqueta"] for o in ops}
                actual = actuales.get(campo)
                indice = valores.index(actual) if actual in valores else 0
                return st.selectbox(
                    titulo,
                    valores,
                    index=indice,
                    format_func=lambda v: etiquetas.get(v, v),
                    key=f"conf_{campo}",
                )

            col_a, col_b = st.columns(2)
            with col_a:
                perfil_sel = _selector("Perfil del destinatario", "perfil_destinatario")
                nicho_sel = _selector("Nicho / sector", "nicho_sector")
            with col_b:
                formato_sel = _selector("Formato de salida", "formato_salida")
                nivel_sel = _selector("Nivel de detalle", "nivel_detalle")

            col_ok, col_cancel = st.columns([2, 1])
            with col_ok:
                if st.button("Ejecutar ▶", type="primary", use_container_width=True):
                    try:
                        consumir_corrida(USUARIO.user_id)
                    except ErrorCuota as exc:
                        st.error(exc.mensaje)
                    else:
                        _reanudar({
                            "confirmado": True,
                            "perfil_destinatario": perfil_sel,
                            "formato_salida": formato_sel,
                            "nicho_sector": nicho_sel,
                            "nivel_detalle": nivel_sel,
                        })
            with col_cancel:
                if st.button("Cancelar", key="conf_cancelar", use_container_width=True):
                    _reanudar({"confirmado": False})

        elif fase == "esperando_aclaracion":
            interrupt_data = st.session_state.interrupt_actual or {}
            st.markdown('<span class="nm-badge pendiente">Necesito una aclaración</span>', unsafe_allow_html=True)
            st.write(
                interrupt_data.get("mensaje_aclaracion")
                or "No encontré ese tema en el documento. ¿Puedes aclarar qué buscas?"
            )
            aclaracion = st.text_area("Tu aclaración (deja vacío para cancelar)", key="texto_aclaracion")
            if st.button("Enviar aclaración"):
                _reanudar({"respuesta": aclaracion.strip()})

        elif fase == "esperando_modificacion":
            values = (
                st.session_state.estado_final.values
                if st.session_state.estado_final is not None
                else {}
            )
            interrupt_data = st.session_state.interrupt_actual or {}
            vueltas = interrupt_data.get("vueltas_modificacion", 0)

            _mostrar_resultado(
                values,
                permitir_modificar=True,
                interrupt_data=interrupt_data,
            )

            instruccion = st.text_area(
                "¿Quieres modificar algo? (deja vacío para terminar)",
                key=f"instruccion_modificacion_{vueltas}",
            )
            if st.button("Enviar"):
                texto = None if _es_vacio(instruccion) else (instruccion or "").strip()[:2000]
                _reanudar({"instruccion": texto})

        elif fase == "fin":
            values = (
                st.session_state.estado_final.values
                if st.session_state.estado_final is not None
                else {}
            )
            _mostrar_resultado(values, permitir_modificar=False)

    with col_timeline:
        st.markdown("#### Actividad en vivo")
        if st.session_state.timeline_nodos:
            if en_resultado:
                columnas = st.columns(3)
                for i, nodo in enumerate(st.session_state.timeline_nodos):
                    with columnas[i % 3]:
                        st.markdown(_html_nodo(nodo), unsafe_allow_html=True)
            else:
                for nodo in st.session_state.timeline_nodos:
                    st.markdown(_html_nodo(nodo), unsafe_allow_html=True)
        elif st.session_state.fase == "inicio":
            st.caption("Escribe un tema para empezar.")