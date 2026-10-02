"""Ilustrador: asigna un icono de la lista permitida a cada flashcard."""

from __future__ import annotations

import logging

from agent_state import AgentState
from src.contracts.formatos import ICONOS_FLASHCARD

log = logging.getLogger(__name__)

ICONIFY = "https://api.iconify.design"
PREFIJO_ICONOS = "lucide"
COLOR_ICONO = "%234FC3F7"  # cian del tema ('#' codificado para la URL)

_PERMITIDOS = frozenset(ICONOS_FLASHCARD)


def _url_icono(nombre: str) -> str:
    return f"{ICONIFY}/{PREFIJO_ICONOS}:{nombre}.svg?color={COLOR_ICONO}&height=96"


async def nodo_ilustrador(state: AgentState) -> dict:
    items = (state.get("contenido_adaptado") or {}).get("items") or []
    imagenes: dict[str, str] = {}
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict) or not it.get("frente"):
            continue
        nombre = str(it.get("icono_busqueda") or "").strip().lower()
        if nombre in _PERMITIDOS:
            imagenes[str(i)] = _url_icono(nombre)
        elif nombre:
            log.info("ilustrador: icono fuera de la lista descartado: %r", nombre)
    return {"imagenes_items": imagenes}