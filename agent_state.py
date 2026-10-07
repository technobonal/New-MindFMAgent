"""
agent_state.py

Estado compartido del grafo de NuevaMente. Adaptado del patrón de
agent_state.py de MCPMULTIAgentsResearch (mismo uso de reducers:
add_messages para mensajes de chat reales, operator.add para listas
de acumulación que no son mensajes).

Los campos marcados con "AGREGADO" ya los usaban grafo.py y los agentes pero no
estaban declarados acá. LangGraph solo guarda las claves declaradas en el estado.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


# --------------------------------------------------------------------------
# Sub-esquemas del contrato de salida (sección 7 del spec)
# --------------------------------------------------------------------------

class Metadatos(TypedDict, total=False):
    perfil_aplicado: str
    formato_generado: str
    tiempo_estimado_estudio_minutos: int
    conceptos_clave: list[str]


class ContenidoAdaptado(TypedDict, total=False):
    titulo: str
    introduccion_contextualizada: str
    # items cambia de forma según formato_salida (Guía Paso a Paso,
    # Flashcards, Quiz, TL;DR, Guion de Clase) -- se deja como list[dict]
    # genérico; el sub-esquema exacto lo valida el prompt del Redactor,
    # no el tipado del estado.
    items: list[dict]


class EvaluacionCalidad(TypedDict, total=False):
    anclaje_fuente_score: float
    claridad_pedagogica: str  # "Alta" | "Media" | "Baja"
    observaciones: str


class AlmacenamientoOCI(TypedDict, total=False):
    bucket: str
    objeto_id: str
    status_upload: str


# --------------------------------------------------------------------------
# Estado principal
# --------------------------------------------------------------------------

class AgentState(TypedDict):
    mensajes: Annotated[list, add_messages]  # mensajes de chat reales
    thread_id: str  # st.user.email una vez esté OAuth (sección 13)

    input_sanitizado: Optional[str]

    # AGREGADO: lo devuelve el nodo de ingesta ({"estado": "ingesta_completada"})
    estado: Optional[str]

    # --- Buscador de Documentos (5to spec) ---
    tema_pedido_chat: Optional[str]
    candidatos_documento: list[str]
    objeto_id_confirmado: Optional[str]
    ejecucion_confirmada: bool

    # --- Parámetros de personalización ---
    # Los resuelve el nodo "parametros" (paso a paso) y se confirman antes de
    # ejecutar. Siempre en forma canónica del contrato (contracts/enums.py).
    # nicho_sector y nivel_detalle arrancan en None = default (General / Estandar).
    tema_consulta: Optional[str]  # tema a verificar dentro del documento (Investigador)
    perfil_destinatario: Optional[str]
    formato_salida: Optional[str]
    nicho_sector: Optional[str]
    nivel_detalle: Optional[str]

    # --- Investigador RAG ---
    fuente_confirmada: Optional[bool]
    chunks_fuente_confirmados: Optional[list[str]]
    mensaje_aclaracion: Optional[str]  # lo que le pregunta al usuario si no matchea
    respuesta_aclaracion_usuario: Optional[str]

    # --- Redactor Pedagógico / Crítico-Revisor ---
    contenido_adaptado: Optional[ContenidoAdaptado]
    metadatos: Optional[Metadatos]
    evaluacion_calidad: Optional[EvaluacionCalidad]
    aprobado: Optional[bool]
    intentos_redactor: int  # contador para el tope de reintentos
    feedback_redactor: Optional[str]  # AGREGADO: feedback del Revisor / error de validación

    # --- Agente Modificador ---
    instruccion_modificacion: Optional[str]  # lo que pide el usuario para cambiar
    vueltas_modificacion: int  # contador para el tope de 5 vueltas por sesión
    intentos_modificador: int  # AGREGADO: correcciones del Modificador por instrucción
    aviso_modificacion: Optional[str]  # AGREGADO: p. ej. "no se pudo aplicar ese cambio"

    # --- Guardado ---
    almacenamiento_oci: Optional[AlmacenamientoOCI]

    # --- Estado general del pipeline ---
    status: str  # "exito" | "exito_con_advertencias" | "error" | "cancelado"
    error: Optional[str]

    imagenes_items: Optional[dict[str, str]]  # {"1": url, ...} (claves str: serializan bien en el checkpoint)