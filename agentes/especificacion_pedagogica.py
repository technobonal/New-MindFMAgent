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


# nivel_detalle: solo modula EXTENSIÓN y DENSIDAD (ortogonal al perfil).
# Claves = valores canónicos de contracts.enums.NivelDetalle.
# BORRADOR: alinear con la redacción exacta del prompt del equipo.
GUIA_NIVEL_DETALLE: dict[str, str] = {
    "Didactico": "Explicaciones más desarrolladas y ejemplos; más apoyo por item.",
    "Estandar": "Extensión equilibrada: cubrir los puntos principales con explicación suficiente.",
    "Profundo": "Más items y mayor densidad, sin agregar datos fuera de los chunks fuente.",
}


def guia_nivel_detalle(nivel: str | Enum) -> str:
    return GUIA_NIVEL_DETALLE[_valor(nivel)]