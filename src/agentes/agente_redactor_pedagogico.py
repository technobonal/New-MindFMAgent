"""Core desacoplado del Redactor Pedagógico, adaptable a un nodo futuro."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Annotated, Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from src.contracts import (
    FORMATOS_IMPLEMENTADOS_MVP,
    MODELO_ITEM_POR_FORMATO,
    FormatoSalida,
    SolicitudAdaptacion,
    validar_pasos_consecutivos,
)
from src.contracts.formatos import ItemPaquete, ItemPasoTutorial
from src.errores import ErrorFormatoNoDisponible, ErrorLLM, ErrorSalidaInvalida
from src.prompts.redactor_pedagogico import construir_prompt_redactor

TextoNoVacio = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class EstadoRedactor(str, Enum):
    GENERACION_CON_EVIDENCIA = "GENERACION_CON_EVIDENCIA"
    EVIDENCIA_INSUFICIENTE = "EVIDENCIA_INSUFICIENTE"


class ChunkFuente(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TextoNoVacio
    texto: TextoNoVacio


class ContenidoRedactor(BaseModel):
    """DTO interno: metadatos de redacción más items tipados por formato."""

    model_config = ConfigDict(extra="forbid")

    titulo: TextoNoVacio
    introduccion_contextualizada: TextoNoVacio
    tiempo_estimado_estudio_minutos: int = Field(gt=0)
    conceptos_clave: list[TextoNoVacio] = Field(min_length=1)
    prerrequisitos: list[str] = Field(default_factory=list)
    items: list[ItemPaquete] = Field(min_length=1)


class RespuestaGenerador(BaseModel):
    """Esquema estructurado que recibe el generador inyectado."""

    model_config = ConfigDict(extra="forbid")

    estado: EstadoRedactor
    contenido: ContenidoRedactor | None = None
    motivo_abstencion: str | None = None

    @model_validator(mode="after")
    def _coherencia_abstencion(self) -> "RespuestaGenerador":
        tiene_contenido = self.contenido is not None
        if (self.estado is EstadoRedactor.GENERACION_CON_EVIDENCIA) != tiene_contenido:
            raise ValueError("El estado y la presencia de contenido deben ser coherentes.")
        if self.estado is EstadoRedactor.EVIDENCIA_INSUFICIENTE and not self.motivo_abstencion:
            self.motivo_abstencion = "La evidencia disponible no permite redactar con respaldo."
        return self


class ResultadoRedactor(BaseModel):
    """Resultado interno sin construir un PaqueteEducativo."""

    model_config = ConfigDict(extra="forbid")

    estado: EstadoRedactor
    contenido: ContenidoRedactor | None = None
    motivo_abstencion: str | None = None

    @model_validator(mode="after")
    def _coherencia_estado(self) -> "ResultadoRedactor":
        if (self.estado is EstadoRedactor.GENERACION_CON_EVIDENCIA) != (self.contenido is not None):
            raise ValueError("El estado y la presencia de contenido deben ser coherentes.")
        return self


class GeneradorEstructurado(Protocol):
    async def generate(
        self, *, prompt: str, output_model: type[BaseModel]
    ) -> Any:
        """Produce una salida estructurada validable contra output_model."""


def _normalizar_chunks(chunks: Sequence[ChunkFuente | Mapping[str, Any]]) -> list[ChunkFuente]:
    try:
        normalizados = [ChunkFuente.model_validate(chunk) for chunk in chunks]
    except (ValidationError, TypeError) as exc:
        raise ErrorSalidaInvalida("Los chunks deben incluir un id y texto válidos.") from exc
    ids = [chunk.id for chunk in normalizados]
    if len(ids) != len(set(ids)):
        raise ErrorSalidaInvalida("Los IDs de chunks deben ser únicos.")
    return normalizados


def _validar_items(
    solicitud: SolicitudAdaptacion,
    items: list[dict[str, Any]],
    ids_chunks: set[str],
) -> list[ItemPaquete]:
    modelo = MODELO_ITEM_POR_FORMATO.get(solicitud.formato_salida)
    if modelo is None:
        raise ErrorSalidaInvalida("No existe modelo de item para el formato solicitado.")

    items_validados = []
    try:
        for bruto in items:
            item = modelo.model_validate(bruto)
            anchors = item.anclaje
            if not anchors:
                raise ValueError("Cada item debe incluir al menos un anclaje.")
            desconocidos = set(anchors) - ids_chunks
            if desconocidos:
                raise ValueError(f"El item referencia chunks inexistentes: {sorted(desconocidos)}.")
            items_validados.append(item)
        if solicitud.formato_salida is FormatoSalida.TUTORIAL:
            validar_pasos_consecutivos(
                [item for item in items_validados if isinstance(item, ItemPasoTutorial)]
            )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ErrorSalidaInvalida(f"La salida del Redactor no cumple el contrato: {exc}") from exc
    return items_validados


async def redactar_pedagogicamente(
    solicitud: SolicitudAdaptacion,
    chunks: Sequence[ChunkFuente | Mapping[str, Any]],
    especificacion_pedagogica: Any,
    generador: GeneradorEstructurado,
    feedback_revisor: Any | None = None,
) -> ResultadoRedactor:
    """Genera contenido respaldado y valida formato e integridad de anchors."""
    if not isinstance(solicitud, SolicitudAdaptacion):
        raise ErrorSalidaInvalida("Se requiere una SolicitudAdaptacion validada.")
    if solicitud.formato_salida not in FORMATOS_IMPLEMENTADOS_MVP:
        raise ErrorFormatoNoDisponible(
            f"El formato '{solicitud.formato_salida.value}' no está disponible en esta fase.",
            campo="formato_salida",
            valores_admitidos=sorted(formato.value for formato in FORMATOS_IMPLEMENTADOS_MVP),
        )

    fuentes = _normalizar_chunks(chunks)
    if not fuentes:
        return ResultadoRedactor(
            estado=EstadoRedactor.EVIDENCIA_INSUFICIENTE,
            motivo_abstencion="No se recibieron chunks de fuente para respaldar la generación.",
        )
    try:
        prompt = construir_prompt_redactor(
            solicitud, fuentes, especificacion_pedagogica, feedback_revisor
        )
    except (TypeError, ValueError) as exc:
        raise ErrorSalidaInvalida(f"No se pudo construir el prompt: {exc}") from exc

    try:
        bruto = await generador.generate(prompt=prompt, output_model=RespuestaGenerador)
    except ErrorLLM:
        raise
    except Exception as exc:
        raise ErrorLLM("Falló la generación estructurada del Redactor.") from exc

    try:
        generado = RespuestaGenerador.model_validate(bruto)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ErrorSalidaInvalida(f"La respuesta estructurada es inválida: {exc}") from exc

    if generado.estado is EstadoRedactor.EVIDENCIA_INSUFICIENTE:
        return ResultadoRedactor(
            estado=generado.estado,
            motivo_abstencion=generado.motivo_abstencion,
        )

    assert generado.contenido is not None  # garantizado por RespuestaGenerador
    items_raw = [item.model_dump(mode="json") for item in generado.contenido.items]
    items = _validar_items(solicitud, items_raw, {chunk.id for chunk in fuentes})
    contenido = generado.contenido.model_copy(update={"items": items})
    return ResultadoRedactor(estado=generado.estado, contenido=contenido)
