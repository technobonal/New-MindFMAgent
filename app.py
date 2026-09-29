"""
app.py

UI de Streamlit para NuevaMente. Primera versión funcional:
- Input de tema por chat (Buscador de Documentos)
- Uploader para subir un documento nuevo directo a fuentes/ en OCI
  (Flujo B) -- queda disponible para el Buscador en la próxima búsqueda
- Selectbox de candidatos cuando el match no es único
- HITL "¿Ejecutar?" y "¿Modificar?" vía interrupt() + Command(resume=...)
- Timeline en vivo de los nodos del grafo vía astream()
- Estética: hero navy oscuro con vórtice de fondo, acento cian con glow,
  título en dos tonos (blanco + gradiente cian) -- referencia visual confirmada

Nota de arquitectura: cada interacción abre una conexión nueva al grafo
(asyncio.run + construir_grafo) y la cierra al terminar -- mismo patrón
que los scripts de prueba ya validados. Streamlit reruns todo el script
en cada interacción, así que esto evita manejar un event loop persistente
entre reruns (una optimización para más adelante, no bloqueante ahora).
"""

import asyncio
import base64
import json

import streamlit as st
from langgraph.types import Command

from grafo import construir_grafo
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado
from seguridad.rate_limiter import RateLimiter

st.set_page_config(page_title="NuevaMente", page_icon="🧠", layout="wide")

# --------------------------------------------------------------------------
# Estética: hero navy + vórtice de fondo + acento cian con glow
# (referencia visual confirmada)
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

/* --- Vórtice de fondo del hero: anillos concéntricos con glow --- */
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

.nm-nodo-titulo {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 2px;
}

.nm-nodo-detalle {
    font-size: 0.8rem;
    color: var(--text-secondary);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Helpers async (cada llamada abre y cierra su propia conexión al grafo)
# --------------------------------------------------------------------------

NOMBRES_NODOS = {
    "buscador_documentos": "Buscador de Documentos",
    "confirmar_ejecucion": "Confirmación de ejecución",
    "ingesta": "Ingesta",
    "validacion": "Validación",
    "supervisor": "Supervisor",
    "investigador": "Investigador RAG",
    "redactor_pedagogico": "Redactor Pedagógico",
    "critico_revisor": "Crítico / Revisor",
    "guardado_final": "Guardado final",
    "confirmar_modificacion": "Confirmación de modificación",
    "modificador": "Modificador",
}


async def _subir_documento(nombre_archivo: str, contenido_bytes: bytes) -> dict:
    """Sube un archivo nuevo a fuentes/ en OCI (Flujo B). No arranca el
    pipeline -- el archivo simplemente queda disponible para que el
    Buscador de Documentos lo encuentre en la próxima búsqueda por tema."""
    contenido_b64 = base64.b64encode(contenido_bytes).decode("ascii")
    async with conectar_mcp() as sesion:
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        resultado = await tools["subir_documento_fuente"].ainvoke({
            "nombre_archivo": nombre_archivo,
            "contenido_base64": contenido_b64,
        })
        texto = extraer_texto_resultado(resultado)
        return json.loads(texto)


async def _iniciar_flujo(tema: str, thread_id: str, contenedor_timeline):
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    grafo = await construir_grafo(st.session_state.rate_limiter)
    config = {"configurable": {"thread_id": thread_id}}
    estado_inicial = {
        "mensajes": [],
        "thread_id": thread_id,
        "tema_pedido_chat": tema,
        "candidatos_documento": [],
        "objeto_id_confirmado": None,
        "ejecucion_confirmada": False,
        "intentos_redactor": 0,
        "vueltas_modificacion": 0,
        "status": "",
    }

    resultado = None
    async for chunk in grafo.astream(estado_inicial, config=config, stream_mode="updates"):
        for nombre_nodo, datos_nodo in chunk.items():
            _pintar_nodo(contenedor_timeline, nombre_nodo, activo=True)
            resultado = datos_nodo

    estado_final = await grafo.aget_state(config)
    await grafo.checkpointer.conn.close()
    return estado_final


async def _resumir_flujo(resume_payload: dict, thread_id: str, contenedor_timeline):
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    grafo = await construir_grafo(st.session_state.rate_limiter)
    config = {"configurable": {"thread_id": thread_id}}

    async for chunk in grafo.astream(Command(resume=resume_payload), config=config, stream_mode="updates"):
        for nombre_nodo, datos_nodo in chunk.items():
            _pintar_nodo(contenedor_timeline, nombre_nodo, activo=True)

    estado_final = await grafo.aget_state(config)
    await grafo.checkpointer.conn.close()
    return estado_final


def _pintar_nodo(contenedor, nombre_nodo: str, activo: bool):
    if nombre_nodo not in NOMBRES_NODOS:
        return
    clase = "activo" if activo else ""
    with contenedor:
        st.markdown(f"""
        <div class="nm-nodo {clase}">
            <div class="nm-nodo-titulo">⚡ {NOMBRES_NODOS[nombre_nodo]}</div>
            <div class="nm-nodo-detalle">completado</div>
        </div>
        """, unsafe_allow_html=True)


def correr(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Estado de sesión
# --------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "sesion-" + str(id(st.session_state))
if "fase" not in st.session_state:
    st.session_state.fase = "inicio"  # inicio | esperando_ejecucion | esperando_modificacion | fin
if "interrupt_actual" not in st.session_state:
    st.session_state.interrupt_actual = None
if "estado_final" not in st.session_state:
    st.session_state.estado_final = None

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.markdown("""
<div class="nm-hero">
    <div class="nm-hero-content">
        <span class="nm-badge">🧠 NuevaMente</span>
        <h1 class="nm-hero-title">
            Contenido educativo
            <span class="linea-acento">generado con IA</span>
        </h1>
        <p class="nm-hero-sub">
            Escribí un tema, elegí la fuente y dejá que el pipeline de agentes
            redacte, revise y guarde el resultado por vos.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

col_form, col_timeline = st.columns([1, 1])

with col_form:
    if st.session_state.fase == "inicio":
        with st.expander("📎 ¿No está el documento que buscás? Subilo acá"):
            archivo_subido = st.file_uploader("Elegí un PDF, Markdown o texto", type=["pdf", "md", "txt"])
            if archivo_subido and st.button("Subir a Object Storage"):
                with st.spinner(f"Subiendo {archivo_subido.name}..."):
                    resultado_subida = correr(_subir_documento(archivo_subido.name, archivo_subido.read()))
                st.success(f"✅ Subido a `{resultado_subida['objeto_id']}` — ya podés buscarlo por tema abajo.")

        tema = st.chat_input("¿Sobre qué tema querés generar contenido?")
        if tema:
            with col_timeline:
                st.markdown("#### Actividad en vivo")
                contenedor = st.container()
            estado = correr(_iniciar_flujo(tema, st.session_state.thread_id, contenedor))
            st.session_state.estado_final = estado
            if estado.tasks and estado.tasks[0].interrupts:
                st.session_state.interrupt_actual = estado.tasks[0].interrupts[0].value
                st.session_state.fase = "esperando_ejecucion"
            st.rerun()

    elif st.session_state.fase == "esperando_ejecucion":
        interrupt_data = st.session_state.interrupt_actual
        candidatos = interrupt_data.get("candidatos_documento", [])
        objeto_id = interrupt_data.get("objeto_id_confirmado")

        if candidatos:
            st.markdown('<span class="nm-badge pendiente">Elegí un documento</span>', unsafe_allow_html=True)
            elegido = st.selectbox("Documentos encontrados", candidatos)
        else:
            st.markdown('<span class="nm-badge completado">Documento encontrado</span>', unsafe_allow_html=True)
            st.write(f"📄 `{objeto_id}`")
            elegido = None

        if st.button("Ejecutar ▶", type="primary"):
            with col_timeline:
                st.markdown("#### Actividad en vivo")
                contenedor = st.container()
            resume_payload = {
                "confirmado": True,
                "objeto_id_confirmado": elegido,
            }
            estado = correr(_resumir_flujo(resume_payload, st.session_state.thread_id, contenedor))
            st.session_state.estado_final = estado
            if estado.tasks and estado.tasks[0].interrupts:
                st.session_state.interrupt_actual = estado.tasks[0].interrupts[0].value
                st.session_state.fase = "esperando_modificacion"
            else:
                st.session_state.fase = "fin"
            st.rerun()

    elif st.session_state.fase == "esperando_modificacion":
        resultado = st.session_state.estado_final.values.get("contenido_adaptado", {})
        st.markdown('<span class="nm-badge completado">Contenido generado</span>', unsafe_allow_html=True)
        st.subheader(resultado.get("titulo", "Sin título"))
        st.write(resultado.get("introduccion_contextualizada", ""))

        instruccion = st.text_area("¿Querés modificar algo? (dejá vacío para terminar)")
        if st.button("Enviar"):
            with col_timeline:
                st.markdown("#### Actividad en vivo")
                contenedor = st.container()
            resume_payload = {"instruccion": instruccion if instruccion.strip() else None}
            estado = correr(_resumir_flujo(resume_payload, st.session_state.thread_id, contenedor))
            st.session_state.estado_final = estado
            if estado.tasks and estado.tasks[0].interrupts:
                st.session_state.interrupt_actual = estado.tasks[0].interrupts[0].value
                st.session_state.fase = "esperando_modificacion"
            else:
                st.session_state.fase = "fin"
            st.rerun()

    elif st.session_state.fase == "fin":
        resultado = st.session_state.estado_final.values.get("contenido_adaptado", {})
        st.markdown('<span class="nm-badge completado">✅ Listo</span>', unsafe_allow_html=True)
        st.subheader(resultado.get("titulo", "Sin título"))
        st.write(resultado.get("introduccion_contextualizada", ""))
        almacenamiento = st.session_state.estado_final.values.get("almacenamiento_oci", {})
        st.caption(f"Guardado en: `{almacenamiento.get('objeto_id', '')}`")

        if st.button("Empezar de nuevo"):
            st.session_state.fase = "inicio"
            st.session_state.interrupt_actual = None
            st.session_state.estado_final = None
            st.rerun()

with col_timeline:
    if st.session_state.fase == "inicio":
        st.markdown("#### Actividad en vivo")
        st.caption("Escribí un tema para empezar.")