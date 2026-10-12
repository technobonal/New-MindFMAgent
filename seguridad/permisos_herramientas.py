"""Permisos mínimos por etapa para las herramientas MCP determinísticas del grafo de NuevaMente.

Solo las etapas determinísticas del grafo usan herramientas MCP. Los agentes con
LLM (supervisor, investigador, redactor_pedagogico, critico_revisor, modificador)
no reciben ninguna: trabajan únicamente con el generador de texto estructurado.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class AccesoHerramientaDenegado(PermissionError):
    """La etapa no tiene permiso para invocar una herramienta MCP."""


HERRAMIENTAS_POR_ETAPA: dict[str, frozenset[str]] = {
    "buscador_documentos": frozenset({"listar_documentos_fuente"}),
    "ingesta": frozenset({"descargar_documento"}),
    "guardado_final": frozenset({"guardar_resultado_formateado"}),
}


def obtener_herramienta_autorizada(
    herramientas: Iterable[Any], *, etapa: str, nombre: str
) -> Any:
    """Resuelve una herramienta MCP solo si está permitida en esa etapa."""
    if nombre not in HERRAMIENTAS_POR_ETAPA.get(etapa, frozenset()):
        raise AccesoHerramientaDenegado(
            f"La etapa {etapa!r} no tiene permiso para la herramienta {nombre!r}."
        )
    for herramienta in herramientas:
        if getattr(herramienta, "name", None) == nombre:
            return herramienta
    raise AccesoHerramientaDenegado(
        f"La herramienta autorizada {nombre!r} no fue expuesta por el servidor MCP."
    )