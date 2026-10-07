"""
agentes/supervisor.py

Extrae el tema de la consulta del usuario (IntencionOut). No genera contenido
educativo ni evalúa si el tema existe en el documento (eso lo hace el
Investigador) -- prompt cerrado en la sección 10 del spec.

Los parámetros de personalización (perfil, formato, nicho, nivel_detalle) YA NO
se infieren acá: los resuelve el nodo "parametros" (agentes/parametros.py), paso
a paso, y el usuario los confirma antes de ejecutar. Si este nodo los devolviera,
pisaría esa elección.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agent_state import AgentState
from llm_client import get_llm
from seguridad.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sos el supervisor de un sistema que transforma documentación técnica en contenido educativo.
Tu única tarea es extraer el tema de la solicitud del usuario y devolver un objeto IntencionOut con:
- tema_consulta: el tema específico que el usuario quiere aprender, en pocas palabras. No incluyas el perfil del destinatario, el formato pedido, el sector ni el nivel de detalle, ni cantidades como "5 preguntas".
No generás contenido educativo vos mismo. No evalúas si el tema existe en el documento — eso lo hace el Investigador.
Si la solicitud no menciona un tema concreto, devolvé la solicitud completa tal cual."""


class IntencionOut(BaseModel):
    tema_consulta: str = Field(
        description=(
            "Tema específico que el usuario quiere aprender, sin perfil, formato, "
            "nicho ni nivel de detalle."
        )
    )


def construir_nodo_supervisor(rate_limiter: RateLimiter):
    """
    Factory: devuelve la función de nodo ya cerrada sobre el
    rate_limiter compartido -- mismo patrón que MCPMULTIAgentsResearch,
    para que construir_grafo() la registre como nodo del StateGraph.
    """

    def nodo_supervisor(state: AgentState) -> dict:
        pedido_usuario = state.get("tema_pedido_chat") or ""
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pedido_usuario},
        ]

        llm = get_llm(mensajes, rate_limiter, temperature=0.1)
        llm_estructurado = llm.with_structured_output(IntencionOut)

        tema_llm: str | None = None
        try:
            intencion: IntencionOut = llm_estructurado.invoke(mensajes)
            tema_llm = (intencion.tema_consulta or "").strip() or None
        except Exception as exc:
            # Structured output falló: no tumbar el grafo; se usa el pedido del chat.
            log.warning("Supervisor: no se pudo extraer el tema (%s); se usa el pedido tal cual.", exc)

        return {"tema_consulta": tema_llm or pedido_usuario or None}

    return nodo_supervisor