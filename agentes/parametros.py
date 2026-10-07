"""
agentes/parametros.py

Asistente de parámetros de NuevaMente (nodo "parametros" del grafo).

El chat pide los parámetros UNO POR UNO, con opciones tocables, hasta completar
lo que falta para ejecutar:

    documento -> perfil -> formato -> (confirmación: nicho y nivel editables)

Reglas (decisiones del equipo):
  - Obligatorios: documento, perfil_destinatario y formato_salida.
  - nicho_sector y nivel_detalle NO se preguntan: entran con General / Estandar
    y se muestran en la confirmación para que el usuario los cambie si quiere.
  - Si el mensaje inicial ya trae perfil, formato, nicho o nivel, se toman de ahí
    y no se vuelve a preguntar (se ven igual en la confirmación).
  - Sin LLM: las preguntas y las opciones las arma el código (cero tokens, sin
    riesgo de inyección). Lo escrito a mano se interpreta con los enums del
    contrato (contracts/enums.py), que ya aceptan alias, tildes y mayúsculas.
  - Formato: solo se ofrecen los implementados en el MVP (FORMATOS_IMPLEMENTADOS_MVP).

Mecánica: un interrupt() por paso dentro de UN solo nodo. LangGraph vuelve a
ejecutar el nodo desde el principio en cada reanudación y entrega las respuestas
ya dadas en orden, por eso todo lo que está antes de los interrupt() es
determinístico (solo lee el estado).

Payload de cada pregunta (lo consume app.py):
    {"tipo": "pedir_parametro", "parametro": "documento" | "perfil_destinatario"
     | "formato_salida", "pregunta": str, "opciones": [{"valor", "etiqueta",
     "ayuda"}], "paso": int, "total": int, "resueltos": {etiqueta: valor},
     "aviso": str | None}
Respuesta esperada (Command(resume=...)):
    {"valor": "<valor canónico o texto escrito>"}  |  {"cancelar": True}
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langgraph.types import interrupt

from agent_state import AgentState
from agentes.especificacion_pedagogica import ESPECIFICACION_POR_PERFIL
from src.contracts.enums import (
    _ALIAS_REGISTRY,  # registro de alias del contrato: fuente única de sinónimos
    FORMATOS_IMPLEMENTADOS_MVP,
    FormatoSalida,
    NichoSector,
    NivelDetalle,
    PerfilDestinatario,
    normalizar,
)

log = logging.getLogger(__name__)

PREFIJO_FUENTES = "fuentes/"

#: Veces que se vuelve a mostrar una pregunta si lo escrito no se reconoce.
MAX_REPREGUNTAS = 2

MENSAJE_SIN_DOCUMENTOS = (
    "No hay documentos disponibles en tu espacio. Subí uno y volvé a intentarlo."
)
MENSAJE_SIN_RESPUESTA_VALIDA = (
    "No pude identificar tu elección. Empezá una consulta nueva y elegí una de las opciones."
)
MENSAJE_AVISO_OPCION = "No reconocí esa opción. Elegí una de las opciones o escribila de nuevo."

CAMPOS_CONFIRMACION = ("perfil_destinatario", "formato_salida", "nicho_sector", "nivel_detalle")

_ENUM_POR_CAMPO = {
    "perfil_destinatario": PerfilDestinatario,
    "formato_salida": FormatoSalida,
    "nicho_sector": NichoSector,
    "nivel_detalle": NivelDetalle,
}

# Etiquetas para pantalla (con tildes). El valor interno es siempre el canónico.
_ETIQUETAS = {
    "perfil_destinatario": {
        "Principiante": "Principiante",
        "Desarrollador": "Desarrollador",
        "Lider Tecnico": "Líder Técnico",
        "Gestor Ejecutivo": "Gestor Ejecutivo",
    },
    "formato_salida": {
        "Tutorial": "Tutorial",
        "Flashcards": "Flashcards",
        "Quiz": "Quiz",
        "Resumen Ejecutivo": "Resumen Ejecutivo",
        "Guion de Clase": "Guion de Clase",
    },
    "nicho_sector": {
        "Fintech": "Fintech",
        "Salud": "Salud",
        "E-commerce": "E-commerce",
        "General": "General",
    },
    "nivel_detalle": {
        "Didactico": "Didáctico",
        "Estandar": "Estándar",
        "Profundo": "Profundo",
    },
}

_AYUDA_FORMATO = {
    "Tutorial": "Guía práctica paso a paso",
    "Flashcards": "Tarjetas de memorización",
    "Quiz": "Preguntas con justificaciones",
    "Resumen Ejecutivo": "TL;DR para decidir",
}

#: Orden de presentación de los formatos (el del documento del Hackathon).
_ORDEN_FORMATOS = ("Tutorial", "Flashcards", "Quiz", "Resumen Ejecutivo")

_DEFAULT_NICHO = NichoSector.GENERAL.value
_DEFAULT_NIVEL = NivelDetalle.ESTANDAR.value


# --------------------------------------------------------------------------
# Etiquetas y opciones
# --------------------------------------------------------------------------

def etiqueta(campo: str, valor: str | None) -> str:
    """Etiqueta de pantalla de un valor canónico."""
    if valor is None:
        return ""
    return _ETIQUETAS.get(campo, {}).get(valor, valor)


def _ayuda(campo: str, valor: str) -> str:
    if campo == "perfil_destinatario":
        espec = ESPECIFICACION_POR_PERFIL.get(valor)
        return espec.nombre_largo if espec else ""
    if campo == "formato_salida":
        return _AYUDA_FORMATO.get(valor, "")
    return ""


def opciones_parametro(campo: str) -> list[dict]:
    """Opciones tocables de un parámetro: [{"valor", "etiqueta", "ayuda"}, ...]."""
    if campo == "formato_salida":
        valores = [v for v in _ORDEN_FORMATOS if FormatoSalida(v) in FORMATOS_IMPLEMENTADOS_MVP]
        valores += sorted(f.value for f in FORMATOS_IMPLEMENTADOS_MVP if f.value not in valores)
    else:
        valores = [m.value for m in _ENUM_POR_CAMPO[campo]]
    return [
        {"valor": v, "etiqueta": etiqueta(campo, v), "ayuda": _ayuda(campo, v)} for v in valores
    ]


def opciones_confirmacion() -> dict[str, list[dict]]:
    """Opciones de los 4 parámetros para los selectores de la pantalla de confirmación."""
    return {campo: opciones_parametro(campo) for campo in CAMPOS_CONFIRMACION}


# --------------------------------------------------------------------------
# Normalización con los enums del contrato
# --------------------------------------------------------------------------

def _canonico(campo: str, valor) -> str | None:
    """Valor canónico del enum del contrato, o None si está vacío o no se reconoce."""
    if valor is None or not str(valor).strip():
        return None
    try:
        return _ENUM_POR_CAMPO[campo](str(valor).strip()).value
    except ValueError:
        return None


def _formato_disponible(valor) -> str | None:
    """Canónico del formato, solo si está implementado en el MVP."""
    canonico = _canonico("formato_salida", valor)
    if canonico and FormatoSalida(canonico) in FORMATOS_IMPLEMENTADOS_MVP:
        return canonico
    return None


def _documento_valido(texto: str, candidatos: list[str]) -> str | None:
    t = (texto or "").strip()
    if t in candidatos:
        return t
    clave = normalizar(t)
    if not clave:
        return None
    for c in candidatos:
        if normalizar(c) == clave:
            return c
    return None


# --------------------------------------------------------------------------
# Detección en el mensaje inicial (determinística, sin LLM)
# --------------------------------------------------------------------------

#: (campo, términos base que NO se usan para detectar). Se excluyen palabras
#: demasiado comunes en español para ser una señal confiable.
#: El orden importa: el formato va primero para "consumir" frases como
#: "resumen ejecutivo" antes de buscar el perfil ("ejecutivo").
_CAMPOS_DETECTABLES = (
    ("formato_salida", frozenset()),
    ("perfil_destinatario", frozenset()),
    ("nicho_sector", frozenset({"general"})),
    ("nivel_detalle", frozenset({"medio", "basico", "avanzado", "estandar"})),
)


def _base(texto: str) -> str:
    """Normaliza y quita la 's' final de cada palabra (plural simple): 'principiantes' -> 'principiante'."""
    return " ".join(
        t[:-1] if len(t) > 3 and t.endswith("s") else t for t in normalizar(texto).split()
    )


@lru_cache(maxsize=None)
def _terminos(enum_cls) -> tuple[tuple[str, str], ...]:
    """(término base, valor canónico) del enum y sus alias, de más largo a más corto."""
    por_termino: dict[str, str] = {}
    for miembro in enum_cls:
        por_termino[_base(miembro.value)] = miembro.value
    for alias, canonico in _ALIAS_REGISTRY.get(enum_cls, {}).items():
        por_termino.setdefault(_base(alias), canonico)
    return tuple(sorted(por_termino.items(), key=lambda kv: len(kv[0]), reverse=True))


def _detectar(enum_cls, texto: str, excluidos: frozenset[str]) -> tuple[str | None, str]:
    """
    Busca términos del enum en `texto` (palabras completas, de más largo a más corto).
    Devuelve (valor, texto sin lo encontrado). Si aparecen valores distintos del mismo
    parámetro, es ambiguo y devuelve None (se preguntará).
    """
    encontrados: set[str] = set()
    for termino, canonico in _terminos(enum_cls):
        if termino in excluidos:
            continue
        patron = rf"(?<!\w){re.escape(termino)}(?!\w)"
        texto, n = re.subn(patron, " ", texto)
        if n:
            encontrados.add(canonico)
    return (next(iter(encontrados)) if len(encontrados) == 1 else None), texto


def detectar_parametros(texto: str | None) -> dict[str, str]:
    """
    Parámetros que el mensaje del usuario ya trae, sin ambigüedad.
    Devuelve {campo: valor_canónico}, p. ej. {"perfil_destinatario": "Principiante"}.
    """
    if not texto or not str(texto).strip():
        return {}
    restante = f" {_base(str(texto))} "
    detectados: dict[str, str] = {}
    for campo, excluidos in _CAMPOS_DETECTABLES:
        valor, restante = _detectar(_ENUM_POR_CAMPO[campo], restante, excluidos)
        if valor:
            detectados[campo] = valor
    return detectados


# --------------------------------------------------------------------------
# Nodo
# --------------------------------------------------------------------------

def _preguntar(
    *,
    parametro: str,
    pregunta: str,
    opciones: list[dict],
    paso: int,
    total: int,
    resueltos: dict[str, str],
    validar,
    aviso: str | None = None,
) -> tuple[str | None, str | None]:
    """
    interrupt() hasta obtener una respuesta válida.
    Devuelve (valor, motivo): motivo es None si hubo valor, "cancelado" si el usuario
    canceló, o "sin_respuesta_valida" si agotó las re-preguntas.
    """
    for _ in range(MAX_REPREGUNTAS + 1):
        respuesta = interrupt({
            "tipo": "pedir_parametro",
            "parametro": parametro,
            "pregunta": pregunta,
            "opciones": opciones,
            "paso": paso,
            "total": total,
            "resueltos": dict(resueltos),
            "aviso": aviso,
        })
        if not isinstance(respuesta, dict):
            respuesta = {"valor": respuesta}
        if respuesta.get("cancelar"):
            return None, "cancelado"
        valor = validar(str(respuesta.get("valor") or ""))
        if valor is not None:
            return valor, None
        aviso = MENSAJE_AVISO_OPCION
    return None, "sin_respuesta_valida"


def _salida_sin_ejecucion(motivo: str) -> dict:
    if motivo == "cancelado":
        return {"ejecucion_confirmada": False, "status": "cancelado", "error": None}
    return {
        "ejecucion_confirmada": False,
        "status": "error",
        "error": MENSAJE_SIN_RESPUESTA_VALIDA,
    }


def nodo_parametros(state: AgentState) -> dict:
    detectados = detectar_parametros(state.get("tema_pedido_chat"))
    objeto_id = state.get("objeto_id_confirmado")
    candidatos = list(state.get("candidatos_documento") or [])

    perfil = _canonico("perfil_destinatario", state.get("perfil_destinatario")) or detectados.get(
        "perfil_destinatario"
    )

    formato = _formato_disponible(state.get("formato_salida"))
    aviso_formato: str | None = None
    if not formato:
        detectado = detectados.get("formato_salida")
        formato = _formato_disponible(detectado)
        if detectado and not formato:
            aviso_formato = (
                f"El formato '{etiqueta('formato_salida', detectado)}' todavía no está "
                "disponible. Elegí uno de estos."
            )

    pendientes = [
        parametro
        for parametro, falta in (
            ("documento", not objeto_id),
            ("perfil_destinatario", not perfil),
            ("formato_salida", not formato),
        )
        if falta
    ]
    total = len(pendientes) + 1  # +1: la pantalla de confirmación

    resueltos: dict[str, str] = {}
    if objeto_id:
        resueltos["Documento"] = str(objeto_id).replace(PREFIJO_FUENTES, "", 1)
    if perfil:
        resueltos["Perfil"] = etiqueta("perfil_destinatario", perfil)
    if formato:
        resueltos["Formato"] = etiqueta("formato_salida", formato)

    # --- Paso: documento -----------------------------------------------------
    if "documento" in pendientes:
        if not candidatos:
            return {"ejecucion_confirmada": False, "status": "error", "error": MENSAJE_SIN_DOCUMENTOS}
        valor, motivo = _preguntar(
            parametro="documento",
            pregunta="¿Con qué documento querés trabajar?",
            opciones=[{"valor": c, "etiqueta": c, "ayuda": ""} for c in candidatos],
            paso=pendientes.index("documento") + 1,
            total=total,
            resueltos=resueltos,
            validar=lambda texto: _documento_valido(texto, candidatos),
        )
        if motivo:
            return _salida_sin_ejecucion(motivo)
        objeto_id = f"{PREFIJO_FUENTES}{valor}"
        resueltos = {"Documento": valor, **resueltos}

    # --- Paso: perfil --------------------------------------------------------
    if "perfil_destinatario" in pendientes:
        valor, motivo = _preguntar(
            parametro="perfil_destinatario",
            pregunta="¿Para quién es el contenido?",
            opciones=opciones_parametro("perfil_destinatario"),
            paso=pendientes.index("perfil_destinatario") + 1,
            total=total,
            resueltos=resueltos,
            validar=lambda texto: _canonico("perfil_destinatario", texto),
        )
        if motivo:
            return _salida_sin_ejecucion(motivo)
        perfil = valor
        resueltos["Perfil"] = etiqueta("perfil_destinatario", perfil)

    # --- Paso: formato -------------------------------------------------------
    if "formato_salida" in pendientes:
        valor, motivo = _preguntar(
            parametro="formato_salida",
            pregunta="¿En qué formato lo querés?",
            opciones=opciones_parametro("formato_salida"),
            paso=pendientes.index("formato_salida") + 1,
            total=total,
            resueltos=resueltos,
            validar=_formato_disponible,
            aviso=aviso_formato,
        )
        if motivo:
            return _salida_sin_ejecucion(motivo)
        formato = valor
        resueltos["Formato"] = etiqueta("formato_salida", formato)

    return {
        "objeto_id_confirmado": objeto_id,
        "candidatos_documento": [],
        "perfil_destinatario": perfil,
        "formato_salida": formato,
        # None = "usar el default"; la confirmación muestra General / Estandar.
        "nicho_sector": _canonico("nicho_sector", state.get("nicho_sector"))
        or detectados.get("nicho_sector"),
        "nivel_detalle": _canonico("nivel_detalle", state.get("nivel_detalle"))
        or detectados.get("nivel_detalle"),
    }


def enrutar_tras_parametros(state: AgentState) -> str:
    return "cancelado" if state.get("status") in ("cancelado", "error") else "ok"


# --------------------------------------------------------------------------
# Confirmación: valores actuales y resolución de lo que edita el usuario
# --------------------------------------------------------------------------

def parametros_para_confirmacion(state: AgentState) -> dict[str, str | None]:
    """Los 4 parámetros tal como se muestran en la confirmación (con defaults en nicho y nivel)."""
    return {
        "perfil_destinatario": _canonico("perfil_destinatario", state.get("perfil_destinatario")),
        "formato_salida": _formato_disponible(state.get("formato_salida")),
        "nicho_sector": _canonico("nicho_sector", state.get("nicho_sector")) or _DEFAULT_NICHO,
        "nivel_detalle": _canonico("nivel_detalle", state.get("nivel_detalle")) or _DEFAULT_NIVEL,
    }


def resolver_confirmacion(state: AgentState, respuesta: dict) -> dict[str, str | None]:
    """
    Valores finales tras la confirmación: lo que el usuario cambió (validado con los
    enums del contrato) o, si no cambió o mandó algo inválido, lo que ya había.
    """
    actuales = parametros_para_confirmacion(state)
    elegidos: dict[str, str | None] = {}
    for campo in CAMPOS_CONFIRMACION:
        crudo = respuesta.get(campo)
        nuevo = _formato_disponible(crudo) if campo == "formato_salida" else _canonico(campo, crudo)
        elegidos[campo] = nuevo or actuales[campo]
    return elegidos