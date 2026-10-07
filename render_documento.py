"""
render_documento.py

Formato "documento" para NuevaMente (hoja clara, título azul oscuro, subtítulos
rojo/naranja, viñetas con etiqueta en negrita), según el ejemplo del PDF del
enunciado del Proyecto 1.

Se aplica a: Resumen Ejecutivo, Tutorial y Quiz (estático, con respuestas al final).
Las Flashcards NO pasan por acá: tipo_documento() devuelve None y app.py sigue
con su render actual.

Este módulo no importa nada de app.py (evita imports circulares) y no usa
Streamlit: solo arma HTML y CSS como texto.

Nota: el HTML se arma en UNA sola línea (sin saltos ni líneas en blanco), porque
st.markdown corta los bloques HTML en la primera línea vacía.
"""

import html
from typing import Optional

# --------------------------------------------------------------------------
# CSS (todo scopeado bajo .nm-doc para no tocar el resto de la app)
# --------------------------------------------------------------------------

CSS_DOCUMENTO = """
.nm-doc {
    --doc-papel: #FFFFFF;
    --doc-tinta: #1D2733;
    --doc-azul: #12294A;
    --doc-acento: #C9432E;
    --doc-suave: #6F7E90;
    --doc-linea: #DCE2EA;
    --doc-fondo-codigo: #F2F5F9;
    background: var(--doc-papel);
    color: var(--doc-tinta);
    font-family: "Segoe UI", "Noto Sans", system-ui, -apple-system, sans-serif;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 860px;
    margin: 8px auto 24px auto;
    padding: 40px 56px 48px 56px;
    border-radius: 6px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
.nm-doc div, .nm-doc p, .nm-doc li, .nm-doc span { color: var(--doc-tinta); }

.nm-doc .doc-cabecera {
    text-align: right;
    font-size: 0.78rem;
    color: var(--doc-suave);
    margin-bottom: 26px;
}
.nm-doc .doc-titulo {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.25;
    color: var(--doc-azul);
    margin: 0 0 18px 0;
}
.nm-doc .doc-seccion {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--doc-acento);
    margin: 28px 0 10px 0;
}
.nm-doc .doc-parrafo { margin: 0 0 10px 0; }
.nm-doc .doc-clave { font-weight: 700; color: var(--doc-azul); }

.nm-doc .doc-sub {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--doc-azul);
    margin: 20px 0 6px 0;
}
.nm-doc .doc-lista { margin: 0 0 6px 0; padding-left: 26px; }
.nm-doc .doc-lista li { margin: 3px 0; }
.nm-doc .doc-lista b { color: var(--doc-azul); }
.nm-doc .doc-lista i { color: var(--doc-tinta); }

.nm-doc .doc-fuente {
    font-size: 0.76rem;
    color: var(--doc-suave);
    margin: 2px 0 0 26px;
    word-break: break-all;
}
.nm-doc .doc-codigo {
    background: var(--doc-fondo-codigo);
    border: 1px solid var(--doc-linea);
    border-radius: 6px;
    padding: 12px 14px;
    margin: 8px 0 10px 0;
    overflow-x: auto;
    font-family: Consolas, "Courier New", monospace;
    font-size: 0.86rem;
    line-height: 1.5;
    white-space: pre;
    color: var(--doc-tinta);
}
.nm-doc .doc-aviso {
    border-left: 4px solid var(--doc-acento);
    background: #FBEFEC;
    padding: 8px 14px;
    margin: 8px 0 10px 0;
}
.nm-doc .doc-mensaje {
    border-left: 4px solid var(--doc-azul);
    padding: 4px 0 4px 14px;
    margin: 0 0 10px 0;
    font-weight: 600;
    color: var(--doc-azul);
}
.nm-doc .doc-opcion { margin: 3px 0 3px 26px; }
.nm-doc .doc-opcion b { color: var(--doc-azul); margin-right: 6px; }
.nm-doc .doc-pie {
    margin-top: 30px;
    padding-top: 12px;
    border-top: 1px solid var(--doc-linea);
    font-size: 0.78rem;
    color: var(--doc-suave);
}
.nm-doc .doc-pie div { color: var(--doc-suave); }

@media (max-width: 800px) {
    .nm-doc { padding: 26px 20px 32px 20px; }
    .nm-doc .doc-titulo { font-size: 1.55rem; }
}
"""

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_FORMATOS = {
    "resumen": "punto_clave",
    "tutorial": "paso_numero",
    "quiz": "pregunta",
}


def _t(valor) -> str:
    """Escapa HTML y convierte saltos de línea en <br> (sin dejar \\n en el HTML)."""
    if valor is None:
        return ""
    texto = html.escape(str(valor).strip())
    return texto.replace("\r\n", "<br>").replace("\n", "<br>")


def _letra(valor) -> str:
    """'b' -> 'B' si es una sola letra; si no, se deja tal cual."""
    texto = str(valor or "").strip()
    return texto.upper() if len(texto) == 1 else texto


def _tipo_item(item: dict) -> str:
    for tipo, clave in _FORMATOS.items():
        if clave in item:
            return tipo
    return ""


def tipo_documento(items: list) -> Optional[str]:
    """
    Devuelve 'resumen' | 'tutorial' | 'quiz' si TODOS los ítems son de ese tipo;
    None en cualquier otro caso (flashcards, mezcla, vacío) -> app.py usa su render actual.
    """
    if not items:
        return None
    tipos = {_tipo_item(i) for i in items if isinstance(i, dict)}
    if len(tipos) == 1:
        tipo = next(iter(tipos))
        return tipo or None
    return None


def _fuente(item: dict) -> str:
    anclaje = [str(a) for a in (item.get("anclaje") or [])]
    if not anclaje:
        return ""
    return f'<div class="doc-fuente">Fuente: {_t(", ".join(anclaje))}</div>'


def _seccion(titulo: str) -> str:
    return f'<div class="doc-seccion">{_t(titulo)}</div>'


def _lista(filas: list[str]) -> str:
    if not filas:
        return ""
    return '<ul class="doc-lista">' + "".join(f"<li>{f}</li>" for f in filas) + "</ul>"


def _cabecera(metadatos: dict) -> str:
    formato = metadatos.get("formato_generado") or "Documento"
    datos = []
    if metadatos.get("perfil_aplicado"):
        datos.append(f"Perfil: {metadatos['perfil_aplicado']}")
    if metadatos.get("nicho_aplicado"):
        datos.append(f"Nicho: {metadatos['nicho_aplicado']}")
    if metadatos.get("nivel_detalle_aplicado"):
        datos.append(f"Nivel: {metadatos['nivel_detalle_aplicado']}")
    if metadatos.get("tiempo_estimado_estudio_minutos"):
        datos.append(f"{metadatos['tiempo_estimado_estudio_minutos']} min")
    resto = f" | {' · '.join(datos)}" if datos else ""
    return f'<div class="doc-cabecera">NuevaMente · {_t(formato)}{_t(resto)}</div>'


def _bloque_prerrequisitos(metadatos: dict) -> str:
    prerreq = metadatos.get("prerrequisitos") or []
    if not prerreq:
        return ""
    return _seccion("Prerrequisitos") + _lista([_t(p) for p in prerreq])


def _bloque_final(resultado: dict, metadatos: dict, evaluacion: dict, almacenamiento: dict,
                  incluir_prerrequisitos: bool) -> str:
    partes = []

    recos = [r for r in (resultado.get("recomendaciones_prioritarias") or []) if r]
    if recos:
        partes.append(_seccion("Recomendaciones prioritarias"))
        partes.append(_lista([_t(r) for r in recos]))

    conceptos = [c for c in (metadatos.get("conceptos_clave") or []) if c]
    if conceptos:
        partes.append(_seccion("Conceptos clave"))
        partes.append(
            '<p class="doc-parrafo">' + ", ".join(f"<i>{_t(c)}</i>" for c in conceptos) + "</p>"
        )

    if incluir_prerrequisitos:
        partes.append(_bloque_prerrequisitos(metadatos))

    pie = []
    if evaluacion:
        metricas = []
        if evaluacion.get("anclaje_fuente_score") is not None:
            metricas.append(f"Anclaje a la fuente: {evaluacion['anclaje_fuente_score']}")
        if evaluacion.get("claridad_pedagogica"):
            metricas.append(f"Claridad: {evaluacion['claridad_pedagogica']}")
        if (evaluacion.get("afirmaciones_evaluadas") is not None
                and evaluacion.get("afirmaciones_soportadas") is not None):
            metricas.append(
                f"Afirmaciones respaldadas: {evaluacion['afirmaciones_soportadas']}"
                f"/{evaluacion['afirmaciones_evaluadas']}"
            )
        if evaluacion.get("reintentos_realizados") is not None:
            metricas.append(f"Reintentos: {evaluacion['reintentos_realizados']}")
        if evaluacion.get("supero_umbral") is not None:
            metricas.append("Umbral superado" if evaluacion["supero_umbral"] else "Umbral no superado")
        if metricas:
            pie.append(f"<div>Evaluación de calidad · {_t(' · '.join(metricas))}</div>")
        if evaluacion.get("observaciones"):
            pie.append(f"<div>Observaciones: {_t(evaluacion['observaciones'])}</div>")
    if almacenamiento.get("objeto_id"):
        pie.append(
            f"<div>Guardado en OCI · bucket {_t(almacenamiento.get('bucket', ''))}"
            f" · {_t(almacenamiento['objeto_id'])}</div>"
        )
    if pie:
        partes.append('<div class="doc-pie">' + "".join(pie) + "</div>")

    return "".join(partes)


# --------------------------------------------------------------------------
# Cuerpo por formato
# --------------------------------------------------------------------------

def _cuerpo_resumen(items: list[dict]) -> str:
    partes = [_seccion("Hallazgos clave")]
    for n, item in enumerate(items, start=1):
        filas = []
        if item.get("implicacion"):
            filas.append(f"<b>Implicación:</b> {_t(item['implicacion'])}")
        if item.get("relevancia_negocio"):
            filas.append(f"<b>Impacto negocio:</b> {_t(item['relevancia_negocio'])}")
        if item.get("recomendacion"):
            filas.append(f"<b>Recomendación:</b> {_t(item['recomendacion'])}")
        partes.append(f'<div class="doc-sub">{n}. {_t(item.get("punto_clave", ""))}</div>')
        partes.append(_lista(filas))
        partes.append(_fuente(item))
    return "".join(partes)


def _cuerpo_tutorial(items: list[dict]) -> str:
    partes = [_seccion("Pasos")]
    for n, item in enumerate(items, start=1):
        numero = item.get("paso_numero", n)
        partes.append(f'<div class="doc-sub">Paso {_t(numero)}. {_t(item.get("titulo_paso", ""))}</div>')
        if item.get("instruccion"):
            partes.append(f'<p class="doc-parrafo">{_t(item["instruccion"])}</p>')
        if item.get("codigo_ejemplo"):
            # &#10; = salto de línea que se respeta dentro del bloque (white-space: pre)
            codigo = html.escape(str(item["codigo_ejemplo"]).strip("\n"))
            codigo = codigo.replace("\r\n", "&#10;").replace("\n", "&#10;")
            partes.append(f'<div class="doc-codigo">{codigo}</div>')
        if item.get("resultado_esperado"):
            partes.append(_lista([f"<b>Resultado esperado:</b> {_t(item['resultado_esperado'])}"]))
        if item.get("advertencia"):
            partes.append(f'<div class="doc-aviso"><b>Advertencia:</b> {_t(item["advertencia"])}</div>')
        partes.append(_fuente(item))
    return "".join(partes)


def _cuerpo_quiz(items: list[dict]) -> str:
    partes = [_seccion("Preguntas")]
    for n, item in enumerate(items, start=1):
        partes.append(f'<div class="doc-sub">{n}. {_t(item.get("pregunta", ""))}</div>')
        for opcion in item.get("opciones") or []:
            if isinstance(opcion, dict):
                partes.append(
                    f'<div class="doc-opcion"><b>{_t(_letra(opcion.get("id")))})</b>'
                    f'{_t(opcion.get("texto", ""))}</div>'
                )

    partes.append(_seccion("Respuestas y justificaciones"))
    for n, item in enumerate(items, start=1):
        filas = [f"<b>Respuesta correcta:</b> {_t(_letra(item.get('respuesta_correcta')))}"]
        if item.get("justificacion"):
            filas.append(f"<b>Justificación:</b> {_t(item['justificacion'])}")
        if item.get("analisis_distractores"):
            filas.append(f"<b>Por qué las otras no:</b> {_t(item['analisis_distractores'])}")
        partes.append(f'<div class="doc-sub">{n}. {_t(item.get("pregunta", ""))}</div>')
        partes.append(_lista(filas))
        partes.append(_fuente(item))
    return "".join(partes)


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------

def html_documento(
    resultado: dict,
    metadatos: dict,
    items: list[dict],
    evaluacion: Optional[dict] = None,
    almacenamiento: Optional[dict] = None,
    status: str = "exito",
) -> str:
    """HTML del documento en una sola línea, listo para st.markdown(..., unsafe_allow_html=True)."""
    tipo = tipo_documento(items)
    if tipo is None:
        raise ValueError("html_documento solo aplica a Resumen Ejecutivo, Tutorial y Quiz")

    evaluacion = evaluacion or {}
    almacenamiento = almacenamiento or {}

    partes = [
        '<div class="nm-doc">',
        _cabecera(metadatos),
        f'<div class="doc-titulo">{_t(resultado.get("titulo") or "Sin título")}</div>',
    ]

    if resultado.get("mensaje_principal"):
        partes.append(_seccion("Mensaje clave"))
        partes.append(f'<div class="doc-mensaje">{_t(resultado["mensaje_principal"])}</div>')

    if resultado.get("introduccion_contextualizada"):
        partes.append(_seccion("Resumen" if tipo == "resumen" else "Introducción"))
        partes.append(f'<p class="doc-parrafo">{_t(resultado["introduccion_contextualizada"])}</p>')

    if tipo == "tutorial":
        partes.append(_bloque_prerrequisitos(metadatos))
        partes.append(_cuerpo_tutorial(items))
    elif tipo == "quiz":
        partes.append(_cuerpo_quiz(items))
    else:
        partes.append(_cuerpo_resumen(items))

    partes.append(
        _bloque_final(resultado, metadatos, evaluacion, almacenamiento,
                      incluir_prerrequisitos=(tipo != "tutorial"))
    )
    partes.append("</div>")
    return "".join(partes)