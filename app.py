"""
app.py

UI de Streamlit para NuevaMente. Primera versión funcional:
- Input de tema por chat (Buscador de Documentos)
- Uploader para subir un documento nuevo directo a fuentes/ en OCI
  (Flujo B) -- queda disponible para el Buscador en la próxima búsqueda
- Selectbox de candidatos cuando el match no es único
- HITL "¿Ejecutar?", "aclaración" y "¿Modificar?" vía interrupt() + Command(resume=...)
- Timeline en vivo de los nodos del grafo vía astream(), con el detalle de
  lo que hizo cada fase y persistente entre reruns
- Salida estructurada (JSON) y render del contenido según el formato
  (Tutorial, Flashcards, Quiz, Resumen Ejecutivo)
- Ficha/reporte con BLUF, hallazgos, recomendaciones, evaluación y
  botones Imprimir / Guardar en OCI / Otra consulta
- Estética: hero navy oscuro con vórtice de fondo, acento cian con glow

Nota de arquitectura: cada interacción abre una conexión nueva al grafo
(asyncio.run + construir_grafo) y la cierra al terminar.
"""

import asyncio
import base64
import html
import json
import re
import uuid
import logging
logging.basicConfig(level=logging.INFO)

import streamlit as st
from langgraph.types import Command
from rag.vectorstore import buscar_chunks_con_id, indexar_documento, calcular_anclaje

from grafo import construir_grafo
from Cliente_agemte import conectar_mcp, obtener_tools_langchain, extraer_texto_resultado
from seguridad.rate_limiter import RateLimiter

st.set_page_config(page_title="NuevaMente", page_icon="🧠", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Helpers async
# --------------------------------------------------------------------------

NOMBRES_NODOS = {
    "buscador_documentos": "Buscador de Documentos",
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
}


async def _subir_documento(nombre_archivo: str, contenido_bytes: bytes) -> dict:
    contenido_b64 = base64.b64encode(contenido_bytes).decode("ascii")
    async with conectar_mcp() as sesion:
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        resultado = await tools["subir_documento_fuente"].ainvoke({
            "nombre_archivo": nombre_archivo,
            "contenido_base64": contenido_b64,
        })
        texto = extraer_texto_resultado(resultado)
        return json.loads(texto)


async def _subir_reporte_generado(nombre_archivo: str, contenido_html: str) -> dict:
    contenido_b64 = base64.b64encode(contenido_html.encode("utf-8")).decode("ascii")
    async with conectar_mcp() as sesion:
        tools = {t.name: t for t in await obtener_tools_langchain(sesion)}
        tool = tools.get("subir_documento_fuente") or tools.get("subir_objeto")
        if not tool:
            raise RuntimeError("No se encontró tool de subida en MCP")
        resultado = await tool.ainvoke({
            "nombre_archivo": nombre_archivo,
            "contenido_base64": contenido_b64,
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


async def _consumir_stream(grafo, entrada, config, contenedor):
    slots = {}
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
    for slot in slots.values():
        slot.empty()


async def _iniciar_flujo(tema: str, thread_id: str, contenedor_timeline):
    grafo = await _construir_grafo_app()
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
    await _consumir_stream(grafo, estado_inicial, config, contenedor_timeline)
    estado_final = await grafo.aget_state(config)
    await grafo.checkpointer.conn.close()
    return estado_final


async def _resumir_flujo(resume_payload: dict, thread_id: str, contenedor_timeline):
    grafo = await _construir_grafo_app()
    config = {"configurable": {"thread_id": thread_id}}
    await _consumir_stream(grafo, Command(resume=resume_payload), config, contenedor_timeline)
    estado_final = await grafo.aget_state(config)
    await grafo.checkpointer.conn.close()
    return estado_final


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
    if datos.get("formato_salida"):
        lineas.append(f"Formato: {datos['formato_salida']}")
    if datos.get("tema_consulta"):
        lineas.append(f"Tema: {datos['tema_consulta']}")
    if not lineas:
        lineas.append("Actualizó: " + ", ".join(datos.keys()))
    return lineas


def _html_nodo(item: dict, activo: bool = False) -> str:
    clase = "activo" if activo else ""
    detalle = "<br>".join(html.escape(str(linea)) for linea in item["detalle"])
    return f"""
    <div class="nm-nodo {clase}">
        <div class="nm-nodo-titulo">⚡ {NOMBRES_NODOS[item['nodo']]}</div>
        <div class="nm-nodo-detalle">{detalle}</div>
    </div>
    """


def _html_nodo_en_curso(nombre_nodo: str) -> str:
    return f"""
    <div class="nm-nodo en-curso">
        <div class="nm-nodo-titulo">⏳ {NOMBRES_NODOS[nombre_nodo]}</div>
        <div class="nm-nodo-detalle">en curso…</div>
    </div>
    """


def _pintar_nodo(contenedor, nombre_nodo: str, activo: bool, datos=None, slot=None):
    if nombre_nodo not in NOMBRES_NODOS:
        return
    item = {"nodo": nombre_nodo, "detalle": _resumen_nodo(datos)}
    st.session_state.timeline_nodos.append(item)
    html_item = _html_nodo(item, activo)
    if slot is not None:
        slot.markdown(html_item, unsafe_allow_html=True)
    else:
        with contenedor:
            st.markdown(html_item, unsafe_allow_html=True)


def correr(coro):
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


def _fase_desde_interrupt(valor) -> str:
    if isinstance(valor, dict):
        tipo = valor.get("tipo")
        if tipo == "confirmar_ejecucion":
            return "esperando_ejecucion"
        if tipo == "confirmar_aclaracion":
            return "esperando_aclaracion"
        if tipo == "confirmar_modificacion":
            return "esperando_modificacion"
        if "mensaje_aclaracion" in valor:
            return "esperando_aclaracion"
        if "vueltas_modificacion" in valor:
            return "esperando_modificacion"
    return "esperando_ejecucion"


def _aplicar_estado(estado):
    st.session_state.estado_final = estado
    st.session_state.imagenes_items = {
        int(k): v
        for k, v in ((estado.values or {}).get("imagenes_items") or {}).items()
    }
    if estado.tasks and estado.tasks[0].interrupts:
        valor = estado.tasks[0].interrupts[0].value
        st.session_state.interrupt_actual = valor
        st.session_state.fase = _fase_desde_interrupt(valor)
    else:
        st.session_state.interrupt_actual = None
        st.session_state.fase = "fin"


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
        if imagen:
            col_img, col_card = st.columns([1, 8])
            col_img.image(imagen, width=48)
        else:
            col_card = st.container()
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

    hallazgos = []
    for n, item in enumerate(items, start=1):
        bloque = [f"<h3>{n}. {esc(item.get('punto_clave', ''))}</h3>"]
        if item.get("implicacion"):
            bloque.append(f"<p><strong>Implicación:</strong> {esc(item['implicacion'])}</p>")
        if item.get("relevancia_negocio"):
            bloque.append(f"<p><strong>Impacto en el negocio:</strong> {esc(item['relevancia_negocio'])}</p>")
        if item.get("recomendacion"):
            bloque.append(f"<p><strong>Recomendación:</strong> {esc(item['recomendacion'])}</p>")
        anclaje = item.get("anclaje") or []
        if anclaje:
            bloque.append(
                f"<p style='color:#666;font-size:0.85em'>Fuente: {esc(', '.join(str(a) for a in anclaje))}</p>"
            )
        hallazgos.append("".join(bloque))

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
  <h2>Hallazgos clave</h2>
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
            help="Descargá el reporte en HTML limpio. Abrilo y usá Ctrl+P / Cmd+P para imprimir o Guardar como PDF.",
        )

    with col2:
        if st.button("☁ Guardar en OCI", use_container_width=True, type="secondary"):
            with st.spinner("Subiendo reporte a Object Storage…"):
                try:
                    res = correr(_subir_reporte_generado(nombre_archivo, html_print))
                    objeto = res.get("objeto_id") or res.get("path") or str(res)
                    st.success(f"✅ Guardado en OCI: `{objeto}`")
                except Exception as e:
                    st.error(f"Error al subir: {e}")

    with col3:
        if st.button("🔄 Otra consulta", use_container_width=True):
            st.session_state.fase = "inicio"
            st.session_state.interrupt_actual = None
            st.session_state.estado_final = None
            st.session_state.timeline_nodos = []
            st.session_state.thread_id = "sesion-" + uuid.uuid4().hex
            st.rerun()


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

if st.session_state.fase in ("esperando_modificacion", "fin"):
    st.toggle("⛶ Pantalla completa", key="pantalla_completa")

if st.session_state.get("pantalla_completa") and st.session_state.fase in ("esperando_modificacion", "fin"):
    col_form = st.container()
    col_timeline = st.expander("Actividad en vivo", expanded=False)
else:
    col_form, col_timeline = st.columns([1, 1])


def _contenedor_timeline():
    with col_timeline:
        st.markdown("#### Actividad en vivo")
        return st.container()


with col_form:
    fase = st.session_state.fase

    if fase == "inicio":
        with st.expander("📎 ¿No está el documento que buscás? Subilo acá"):
            archivo_subido = st.file_uploader("Elegí un PDF, Markdown o texto", type=["pdf", "md", "txt"])
            if archivo_subido and st.button("Subir a Object Storage"):
                with st.spinner(f"Subiendo {archivo_subido.name}..."):
                    resultado_subida = correr(_subir_documento(archivo_subido.name, archivo_subido.read()))
                st.success(f"✅ Subido a `{resultado_subida['objeto_id']}` — ya podés buscarlo por tema abajo.")

        tema = st.chat_input("¿Sobre qué tema querés generar contenido?")
        if tema:
            st.session_state.timeline_nodos = []
            contenedor = _contenedor_timeline()
            estado = correr(_iniciar_flujo(tema, st.session_state.thread_id, contenedor))
            _aplicar_estado(estado)
            st.rerun()

    elif fase == "esperando_ejecucion":
        interrupt_data = st.session_state.interrupt_actual or {}
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
            contenedor = _contenedor_timeline()
            resume_payload = {
                "confirmado": True,
                "objeto_id_confirmado": elegido,
            }
            estado = correr(_resumir_flujo(resume_payload, st.session_state.thread_id, contenedor))
            _aplicar_estado(estado)
            st.rerun()

    elif fase == "esperando_aclaracion":
        interrupt_data = st.session_state.interrupt_actual or {}
        st.markdown('<span class="nm-badge pendiente">Necesito una aclaración</span>', unsafe_allow_html=True)
        st.write(
            interrupt_data.get("mensaje_aclaracion")
            or "No encontré ese tema en el documento. ¿Podés aclarar qué buscás?"
        )
        aclaracion = st.text_area("Tu aclaración (dejá vacío para cancelar)", key="texto_aclaracion")
        if st.button("Enviar aclaración"):
            contenedor = _contenedor_timeline()
            resume_payload = {"respuesta": aclaracion.strip()}
            estado = correr(_resumir_flujo(resume_payload, st.session_state.thread_id, contenedor))
            _aplicar_estado(estado)
            st.rerun()

    elif fase == "esperando_modificacion":
        values = st.session_state.estado_final.values if st.session_state.estado_final is not None else {}
        resultado = values.get("contenido_adaptado") or {}
        metadatos = values.get("metadatos") or {}
        evaluacion = values.get("evaluacion_calidad") or {}
        almacenamiento = values.get("almacenamiento_oci") or {}
        status = values.get("status") or "exito"
        interrupt_data = st.session_state.interrupt_actual or {}
        vueltas = interrupt_data.get("vueltas_modificacion", 0)

        st.markdown('<span class="nm-badge completado">Contenido generado</span>', unsafe_allow_html=True)
        _renderizar_contenido(resultado, metadatos, evaluacion, almacenamiento, status)
        _botones_accion_reporte(resultado, metadatos, evaluacion, almacenamiento, status)
        _mostrar_json_crudo(values)

        if interrupt_data.get("aviso"):
            st.warning(interrupt_data["aviso"])

        instruccion = st.text_area(
            "¿Querés modificar algo? (dejá vacío para terminar)",
            key=f"instruccion_modificacion_{vueltas}",
        )
        if st.button("Enviar"):
            contenedor = _contenedor_timeline()
            resume_payload = {"instruccion": None if _es_vacio(instruccion) else instruccion.strip()}
            estado = correr(_resumir_flujo(resume_payload, st.session_state.thread_id, contenedor))
            _aplicar_estado(estado)
            st.rerun()

    elif fase == "fin":
        values = st.session_state.estado_final.values if st.session_state.estado_final is not None else {}
        resultado = values.get("contenido_adaptado") or {}
        metadatos = values.get("metadatos") or {}
        evaluacion = values.get("evaluacion_calidad") or {}
        almacenamiento = values.get("almacenamiento_oci") or {}
        status = values.get("status") or "fin"
        error = values.get("error")

        st.markdown('<span class="nm-badge completado">✅ Listo</span>', unsafe_allow_html=True)

        if error:
            st.warning(f"El flujo terminó con error: {error}")
        elif not resultado:
            st.info(f"Flujo terminado (status: `{status}`). No hay contenido para mostrar.")
        else:
            _renderizar_contenido(resultado, metadatos, evaluacion, almacenamiento, status)
            _botones_accion_reporte(resultado, metadatos, evaluacion, almacenamiento, status)
            _mostrar_json_crudo(values)

        if almacenamiento.get("objeto_id"):
            st.caption(f"Guardado en: `{almacenamiento.get('objeto_id', '')}`")
        else:
            st.caption("Sin objeto guardado en OCI (el flujo no pasó por guardado final o falló la carga).")

with col_timeline:
    st.markdown("#### Actividad en vivo")
    if st.session_state.timeline_nodos:
        for nodo in st.session_state.timeline_nodos:
            st.markdown(_html_nodo(nodo), unsafe_allow_html=True)
    elif st.session_state.fase == "inicio":
        st.caption("Escribí un tema para empezar.")