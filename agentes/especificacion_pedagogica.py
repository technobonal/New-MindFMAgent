"""
agentes/especificacion_pedagogica.py

Fuente única de verdad para Redactor y Revisor: tabla fija perfil -> spec
pedagógica (sin LLM, sin llamadas a Groq). El Revisor usa la MISMA tabla para
"¿se ajusta al perfil?", así ambos miden lo mismo.

Los enums NO se redefinen acá: se usan los del contrato del equipo
(PerfilDestinatario, NivelDetalle en contracts/enums.py). Este módulo es
agnóstico a la ruta de import: las tablas están indexadas por el VALOR
canónico (str), y las funciones aceptan tanto el enum como el str.

Uso esperado (normalizar primero con el enum del contrato):
    perfil = PerfilDestinatario(state["perfil_destinatario"])  # ValueError si no existe
    bloque = especificacion_para_prompt(perfil)

nivel_detalle
-------------
Modula SOLO extensión y densidad (ortogonal al perfil). Cada nivel tiene:
  - una guía textual (GUIA_NIVEL_DETALLE) que usa el prompt del Redactor,
  - un rango de items (RANGO_ITEMS_POR_NIVEL), con tope por formato
    (TOPE_ITEMS_POR_FORMATO) para respetar límites propios de cada formato.
El Revisor usa criterio_nivel_para_revisor() para medir lo mismo que el Redactor.
Una cantidad pedida explícitamente por el usuario siempre manda sobre el rango.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _valor(x: str | Enum) -> str:
    return x.value if isinstance(x, Enum) else x


@dataclass(frozen=True)
class EspecificacionPedagogica:
    nombre_largo: str  # para prompts; el canónico del contrato es el corto
    bloom: tuple[str, ...]
    andamiaje: str
    registro: str
    foco: str
    verbos: tuple[str, ...]


# Claves = valores canónicos de contracts.enums.PerfilDestinatario
ESPECIFICACION_POR_PERFIL: dict[str, EspecificacionPedagogica] = {
    "Principiante": EspecificacionPedagogica(
        nombre_largo="Principiante / Transición de Carrera",
        bloom=("Recordar", "Comprender"),
        andamiaje="Alto: define cada término, usa analogías, pasos chicos",
        registro="Cercano, sin jerga (la jerga se define al usarla)",
        foco="Qué es y para qué sirve; el porqué antes del cómo",
        verbos=("identificar", "describir", "explicar", "reconocer"),
    ),
    "Desarrollador": EspecificacionPedagogica(
        nombre_largo="Desarrollador Junior / Semi Senior",
        bloom=("Aplicar", "Analizar"),
        andamiaje="Medio: asume fundamentos, explica las decisiones",
        registro="Técnico y directo",
        foco="Cómo hacerlo; errores comunes",
        verbos=("implementar", "configurar", "ejecutar", "depurar", "comparar"),
    ),
    "Lider Tecnico": EspecificacionPedagogica(
        nombre_largo="Líder Técnico / Arquitecto",
        bloom=("Analizar", "Evaluar"),
        andamiaje="Bajo",
        registro="Técnico denso",
        foco="Trade-offs, decisiones de diseño, riesgos",
        verbos=("evaluar", "justificar", "diseñar", "priorizar"),
    ),
    "Gestor Ejecutivo": EspecificacionPedagogica(
        nombre_largo="Gestor / Ejecutivo (No Técnico)",
        bloom=("Comprender", "Evaluar"),  # Evaluar = evaluar impacto
        andamiaje="Medio, sin detalle técnico",
        registro="Negocio, conciso",
        foco="Impacto, costo, riesgo, decisión",
        verbos=("decidir", "priorizar", "valorar", "anticipar"),
    ),
}


def obtener_especificacion(perfil: str | Enum) -> EspecificacionPedagogica:
    return ESPECIFICACION_POR_PERFIL[_valor(perfil)]


def especificacion_para_prompt(perfil: str | Enum) -> str:
    """Bloque de texto para el prompt del Redactor (y del Modificador)."""
    s = obtener_especificacion(perfil)
    return (
        f"Perfil del destinatario: {s.nombre_largo}\n"
        f"- Niveles de Bloom a apuntar: {', '.join(s.bloom)}\n"
        f"- Andamiaje: {s.andamiaje}\n"
        f"- Registro: {s.registro}\n"
        f"- Foco: {s.foco}\n"
        f"- Verbos de acción preferidos: {', '.join(s.verbos)}"
    )


def criterio_para_revisor(perfil: str | Enum) -> str:
    """Criterio '¿se ajusta al perfil?' del Revisor, desde la MISMA tabla."""
    s = obtener_especificacion(perfil)
    return (
        f"El contenido se ajusta al perfil '{s.nombre_largo}' si: "
        f"apunta a Bloom {', '.join(s.bloom)}; andamiaje {s.andamiaje.lower()}; "
        f"registro {s.registro.lower()}; foco en {s.foco.lower()}; "
        f"y usa verbos como {', '.join(s.verbos)}."
    )


# ---------------------------------------------------------------------------
# nivel_detalle: solo modula EXTENSIÓN y DENSIDAD (ortogonal al perfil).
# Claves = valores canónicos de contracts.enums.NivelDetalle.
# ---------------------------------------------------------------------------

GUIA_NIVEL_DETALLE: dict[str, str] = {
    "Didactico": (
        "Cobertura acotada y desarrollo amplio de cada item: explicaciones completas, "
        "ejemplos tomados de los chunks y ningún paso intermedio sin explicar si el "
        "documento lo explica. Prioriza claridad sobre cobertura."
    ),
    "Estandar": (
        "Cobertura equilibrada: cubre los puntos principales, una idea por item, con "
        "explicación suficiente y sin ampliaciones."
    ),
    "Profundo": (
        "Mayor cobertura y densidad: más items, e incluye los matices, condiciones, "
        "excepciones y relaciones entre conceptos que el documento menciona. Cada item "
        "es más denso y conciso. Sin agregar datos fuera de los chunks fuente."
    ),
}

#: (mínimo, máximo) de items por nivel. Una cantidad explícita del usuario manda.
RANGO_ITEMS_POR_NIVEL: dict[str, tuple[int, int]] = {
    "Didactico": (3, 5),
    "Estandar": (4, 7),
    "Profundo": (6, 10),
}

#: Tope duro por formato (clave = valor canónico de contracts.enums.FormatoSalida).
#: Resumen Ejecutivo (BLUF) admite como máximo 7 hallazgos.
TOPE_ITEMS_POR_FORMATO: dict[str, int] = {
    "Resumen Ejecutivo": 7,
}


def guia_nivel_detalle(nivel: str | Enum) -> str:
    return GUIA_NIVEL_DETALLE[_valor(nivel)]


def rango_items(nivel: str | Enum, formato: str | Enum | None = None) -> tuple[int, int]:
    """
    Rango (mínimo, máximo) de items para un nivel, recortado por el tope del
    formato si existe. Ej.: Profundo + Resumen Ejecutivo -> (6, 7).
    """
    minimo, maximo = RANGO_ITEMS_POR_NIVEL[_valor(nivel)]
    if formato is not None:
        tope = TOPE_ITEMS_POR_FORMATO.get(_valor(formato))
        if tope is not None:
            maximo = min(maximo, tope)
            minimo = min(minimo, maximo)
    return minimo, maximo


def criterio_nivel_para_revisor(nivel: str | Enum, formato: str | Enum | None = None) -> str:
    """Criterio '¿cumple el nivel de detalle pedido?' del Revisor, desde la MISMA tabla."""
    minimo, maximo = rango_items(nivel, formato)
    return (
        f"El contenido cumple el nivel de detalle '{_valor(nivel)}' si: "
        f"{guia_nivel_detalle(nivel)} "
        f"La cantidad de items esperada es de {minimo} a {maximo}, salvo que el usuario "
        f"haya pedido una cantidad exacta o que los chunks no alcancen para el mínimo "
        f"(en ese caso es preferible menos items antes que items inventados)."
    )