"""
agentes/supervisor.py

Clasifica la intención del usuario (IntencionOut): tema, perfil del
destinatario, formato pedagógico, nicho/sector y nivel de detalle.
No genera contenido educativo, no evalúa si el tema existe en el
documento (eso lo hace el Investigador) -- prompt cerrado en la
sección 10 del spec.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_state import AgentState
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

SYSTEM_PROMPT = """Sos el supervisor de un sistema que transforma documentación técnica en contenido educativo.
Tu única tarea es clasificar la solicitud del usuario y devolver un objeto IntencionOut con:
- tema_consulta: el tema específico que el usuario quiere aprender (extraído de su pedido)
- perfil_destinatario, formato_salida, nicho_sector, nivel_detalle: tal como los indicó el usuario
No generás contenido educativo vos mismo. No evalúas si el tema existe en el documento — eso lo hace el Investigador.
Si falta alguno de los 4 parámetros, marcalo como null y no lo inventes."""


class IntencionOut(BaseModel):
    tema_consulta: str | None = Field(default=None)
    perfil_destinatario: str | None = Field(
        default=None,
        description=(
            "Uno de: 'Principiante/Transición de Carrera', "
            "'Desarrollador Junior/Semi Senior', 'Líder Técnico/Arquitecto', "
            "'Gestor/Ejecutivo No Técnico'"
        ),
    )
    formato_salida: str | None = Field(
        default=None,
        description=(
            "Uno de: 'Guía Práctica Paso a Paso', 'Flashcards', "
            "'Quiz Interactivo con Justificaciones', 'Resumen Ejecutivo (TL;DR)', "
            "'Guion de Clase/Video'"
        ),
    )
    nicho_sector: str | None = Field(
        default=None,
        description="Uno de: 'Fintech', 'Salud', 'E-commerce', 'General'",
    )
    nivel_detalle: str | None = Field(default=None)


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
        intencion: IntencionOut = llm_estructurado.invoke(mensajes)

        # Regla de práctica ya acordada: si el LLM no extrajo tema_consulta,
        # usar tema_pedido_chat como default -- son campos independientes
        # (ver justificación completa charlada en el 5to spec).
        tema_final = intencion.tema_consulta or state.get("tema_pedido_chat")

        return {
            "tema_consulta": tema_final,
            "perfil_destinatario": intencion.perfil_destinatario,
            "formato_salida": intencion.formato_salida,
            "nicho_sector": intencion.nicho_sector,
            "nivel_detalle": intencion.nivel_detalle,
        }

    return nodo_supervisor