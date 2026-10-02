"""
agentes/supervisor.py

Clasifica la intención del usuario (IntencionOut): tema, perfil del
destinatario, formato pedagógico, nicho/sector y nivel de detalle.
No genera contenido educativo, no evalúa si el tema existe en el
documento (eso lo hace el Investigador) -- prompt cerrado en la
sección 10 del spec.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_state import AgentState
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

# Valores canónicos = los que valida el Redactor (una sola fuente de verdad).
PerfilDestinatario = Literal[
    "Principiante",
    "Desarrollador",
    "Lider Tecnico",
    "Gestor Ejecutivo",
]
FormatoSalida = Literal[
    "Flashcards",
    "Tutorial",
    "Quiz",
    "Resumen Ejecutivo",
    "Guion de Clase",
]
NichoSector = Literal["Fintech", "Salud", "E-commerce", "General"]

# Aliases que el LLM o el spec largo pueden devolver → valor canónico.
PERFIL_MAP = {
    "principiante": "Principiante",
    "principiante/transición de carrera": "Principiante",
    "principiante/transicion de carrera": "Principiante",
    "desarrollador": "Desarrollador",
    "desarrollador junior/semi senior": "Desarrollador",
    "desarrollador junior": "Desarrollador",
    "lider tecnico": "Lider Tecnico",
    "líder técnico": "Lider Tecnico",
    "lider técnico": "Lider Tecnico",
    "líder técnico/arquitecto": "Lider Tecnico",
    "lider tecnico/arquitecto": "Lider Tecnico",
    "gestor ejecutivo": "Gestor Ejecutivo",
    "gestor/ejecutivo no técnico": "Gestor Ejecutivo",
    "gestor/ejecutivo no tecnico": "Gestor Ejecutivo",
}

FORMATO_MAP = {
    "flashcards": "Flashcards",
    "tutorial": "Tutorial",
    "guía práctica paso a paso": "Tutorial",
    "guia practica paso a paso": "Tutorial",
    "quiz": "Quiz",
    "quiz interactivo con justificaciones": "Quiz",
    "resumen ejecutivo": "Resumen Ejecutivo",
    "resumen ejecutivo (tl;dr)": "Resumen Ejecutivo",
    "guion de clase": "Guion de Clase",
    "guión de clase": "Guion de Clase",
    "guion de clase/video": "Guion de Clase",
    "guión de clase/video": "Guion de Clase",
}

NICHO_MAP = {
    "fintech": "Fintech",
    "salud": "Salud",
    "e-commerce": "E-commerce",
    "ecommerce": "E-commerce",
    "general": "General",
}

SYSTEM_PROMPT = """Sos el supervisor de un sistema que transforma documentación técnica en contenido educativo.
Tu única tarea es clasificar la solicitud del usuario y devolver un objeto IntencionOut con:
- tema_consulta: el tema específico que el usuario quiere aprender (extraído de su pedido)
- perfil_destinatario: EXACTAMENTE uno de: Principiante | Desarrollador | Lider Tecnico | Gestor Ejecutivo
- formato_salida: EXACTAMENTE uno de: Flashcards | Tutorial | Quiz | Resumen Ejecutivo | Guion de Clase
- nicho_sector: EXACTAMENTE uno de: Fintech | Salud | E-commerce | General
- nivel_detalle: lo que indico el usuario, o null
No generás contenido educativo vos mismo. No evalúas si el tema existe en el documento — eso lo hace el Investigador.
Si el usuario no indicó perfil, formato o nicho, devolvélos como null (no inventes).
Usá los strings EXACTOS de arriba (sin tildes en "Lider Tecnico", sin variantes largas)."""


class IntencionOut(BaseModel):
    tema_consulta: str | None = Field(default=None)
    perfil_destinatario: PerfilDestinatario | None = Field(
        default=None,
        description="Uno de: Principiante, Desarrollador, Lider Tecnico, Gestor Ejecutivo",
    )
    formato_salida: FormatoSalida | None = Field(
        default=None,
        description="Uno de: Flashcards, Tutorial, Quiz, Resumen Ejecutivo, Guion de Clase",
    )
    nicho_sector: NichoSector | None = Field(
        default=None,
        description="Uno de: Fintech, Salud, E-commerce, General",
    )
    nivel_detalle: str | None = Field(default=None)


def normalizar_perfil(v: str | None) -> str | None:
    if v is None or not str(v).strip():
        return None
    key = str(v).strip().lower()
    return PERFIL_MAP.get(key)  # si no está en el mapa, el caller pone default


def normalizar_formato(v: str | None) -> str | None:
    if v is None or not str(v).strip():
        return None
    key = str(v).strip().lower()
    return FORMATO_MAP.get(key, "Tutorial")


def normalizar_nicho(v: str | None) -> str | None:
    if v is None or not str(v).strip():
        return None
    key = str(v).strip().lower()
    return NICHO_MAP.get(key, "General")


def construir_nodo_supervisor(rate_limiter: RateLimiter):
    """
    Factory: devuelve la función de nodo ya cerrada sobre el
    rate_limiter compartido -- mismo patrón que MCPMULTIAgentsResearch,
    para que construir_grafo() la registre como nodo del StateGraph.
    """

    def nodo_supervisor(state: AgentState) -> dict:
        pedido_usuario = state.get("tema_pedido_chat") or ""
        # TODO: cuando exista el formulario de parámetros en app.py
        # (perfil/formato/nicho/nivel, sección 8 del spec), el pedido
        # completo del usuario debería armarse acá con esos 4 valores
        # explícitos en vez de que el LLM los infiera solo del tema.
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pedido_usuario},
        ]

        llm = get_llm(mensajes, rate_limiter, temperature=0.1)
        llm_estructurado = llm.with_structured_output(IntencionOut)

        try:
            intencion: IntencionOut = llm_estructurado.invoke(mensajes)
            perfil = normalizar_perfil(intencion.perfil_destinatario)
            formato = normalizar_formato(intencion.formato_salida)
            nicho = normalizar_nicho(intencion.nicho_sector)
            tema_llm = intencion.tema_consulta
            nivel = intencion.nivel_detalle
        except Exception:
            # Structured output falló (enum viejo / basura del modelo):
            # no tumbar el grafo; defaults seguros y tema del chat.
            perfil, formato, nicho, tema_llm, nivel = None, None, None, None, None

        tema_final = tema_llm or state.get("tema_pedido_chat")

        return {
            "tema_consulta": tema_final,
            # Defaults de producto si el usuario no especificó nada:
            # el Redactor exige valores del enum, no null.
            "perfil_destinatario": perfil or "Desarrollador",
            "formato_salida": formato or "Tutorial",
            "nicho_sector": nicho or "General",
            "nivel_detalle": nivel,
        }

    return nodo_supervisor