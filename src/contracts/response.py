"""
NuevaMente — Contrato de salida: el paquete educativo.

Reproduce la estructura del Ejemplo de Respuesta oficial (p.4-5) y la amplía
con lo que el Decision Gate resolvió:

  A-05  metadatos.nicho_aplicado          (el ejemplo lo omitia)
  A-06  metadatos.nivel_detalle_aplicado
  A-09  metadatos.prerrequisitos          (exigido en p.2, ausente del ejemplo)
  A-11  almacenamiento_oci.objeto_documento_original  (se persisten DOS artefactos)
  D-05  status = exito | exito_con_advertencias | error
  D-03  claridad_pedagogica = "Requiere revision" si la fidelidad quedo baja

Regla de compatibilidad: NINGÚN campo del ejemplo oficial cambió de nombre,
tipo ni significado. Todo lo nuevo es aditivo. El ejemplo de la p.5 valida
contra este contrato tal cual está escrito en el documento.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .enums import (
    ClaridadPedagogica,
    FormatoSalida,
    NichoSector,
    NivelDetalle,
    PerfilDestinatario,
    StatusOperacion,
    StatusUpload,
)
from .formatos import (
    MODELO_ITEM_POR_FORMATO,
    ItemPasoTutorial,
    validar_pasos_consecutivos,
)

TextoNoVacio = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


# =============================================================================
# metadatos
# =============================================================================


class MetadatosAprendizaje(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- campos del ejemplo oficial (p.5) ---
    perfil_aplicado: PerfilDestinatario
    formato_generado: FormatoSalida
    tiempo_estimado_estudio_minutos: int = Field(gt=0)
    conceptos_clave: list[TextoNoVacio] = Field(min_length=1)

    # --- añadidos por el Decision Gate ---
    nicho_aplicado: NichoSector = Field(
        default=NichoSector.GENERAL, description="A-05: el ejemplo oficial lo omitia."
    )
    nivel_detalle_aplicado: NivelDetalle = Field(
        default=NivelDetalle.ESTANDAR, description="A-06."
    )
    prerrequisitos: list[str] = Field(
        default_factory=list,
        description="A-09: exigido en p.2, ausente del ejemplo de la p.5.",
    )


# =============================================================================
# contenido_adaptado
# =============================================================================


class ContenidoAdaptado(BaseModel):
    """
    `titulo` e `introduccion_contextualizada` existen en todos los formatos
    (están en el ejemplo oficial). `items` cambia de forma según el formato:
    su validación se hace en PaqueteEducativo, que es quien conoce el formato.
    """

    model_config = ConfigDict(extra="forbid")

    titulo: TextoNoVacio
    introduccion_contextualizada: TextoNoVacio
    items: list[dict[str, Any]] = Field(min_length=1)


# =============================================================================
# evaluacion_calidad
# =============================================================================


class EvaluacionCalidad(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anclaje_fuente_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Decision D-02: proporcion de afirmaciones soportadas por los "
        "chunks, segun el agente verificador. Las parcialmente soportadas cuentan 0.5.",
    )
    claridad_pedagogica: ClaridadPedagogica
    observaciones: TextoNoVacio

    # --- trazabilidad del mecanismo (decision D-03) ---
    afirmaciones_evaluadas: int = Field(default=0, ge=0)
    afirmaciones_soportadas: int = Field(default=0, ge=0)
    reintentos_realizados: int = Field(default=0, ge=0)
    supero_umbral: bool = Field(
        default=True, description="False si quedo bajo el umbral tras el reintento."
    )

    @model_validator(mode="after")
    def _conteos_coherentes(self) -> "EvaluacionCalidad":
        if self.afirmaciones_soportadas > self.afirmaciones_evaluadas:
            raise ValueError(
                "afirmaciones_soportadas no puede superar a afirmaciones_evaluadas."
            )
        return self


# =============================================================================
# almacenamiento_oci
# =============================================================================


class AlmacenamientoOci(BaseModel):
    """
    A-11: el ejemplo oficial reporta un solo `objeto_id`, pero se persisten DOS
    artefactos (documento original + JSON del paquete). Se conserva `objeto_id`
    con el significado del ejemplo —el JSON generado— y se añade el otro.
    """

    model_config = ConfigDict(extra="forbid")

    bucket: str
    objeto_id: str = Field(description="Objeto del JSON generado. Campo del ejemplo oficial.")
    status_upload: StatusUpload
    objeto_documento_original: str | None = Field(
        default=None, description="A-11: el segundo artefacto persistido."
    )
    detalle_fallo: str | None = Field(
        default=None, description="Motivo legible cuando status_upload es 'fallido'."
    )


# =============================================================================
# El paquete completo
# =============================================================================


class PaqueteEducativo(BaseModel):
    """Respuesta de la acción principal. Estructura del ejemplo oficial (p.4-5)."""

    model_config = ConfigDict(extra="forbid")

    status: StatusOperacion = StatusOperacion.EXITO
    metadatos: MetadatosAprendizaje
    contenido_adaptado: ContenidoAdaptado
    evaluacion_calidad: EvaluacionCalidad
    almacenamiento_oci: AlmacenamientoOci

    # --- trazabilidad interna, no estaba en el ejemplo ---
    advertencias: list[str] = Field(default_factory=list)
    document_id: str | None = None
    generado_en: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def _items_coinciden_con_el_formato(self) -> "PaqueteEducativo":
        """
        Aquí se cumple FR-OUT-05: `items[]` debe validar contra el esquema
        del formato que se pidió, no contra "algún" esquema.

        Es el punto donde el contrato deja de ser una promesa y pasa a ser
        una garantía.
        """
        formato = self.metadatos.formato_generado
        modelo = MODELO_ITEM_POR_FORMATO.get(formato)
        if modelo is None:
            raise ValueError(f"No hay esquema de items definido para el formato {formato}.")

        validados = []
        for i, bruto in enumerate(self.contenido_adaptado.items):
            try:
                validados.append(modelo.model_validate(bruto))
            except Exception as exc:
                raise ValueError(
                    f"El item {i} no cumple el esquema de '{formato.value}': {exc}"
                ) from exc

        if formato is FormatoSalida.TUTORIAL:
            validar_pasos_consecutivos([v for v in validados if isinstance(v, ItemPasoTutorial)])

        return self

    @model_validator(mode="after")
    def _status_coherente(self) -> "PaqueteEducativo":
        """
        Decisión D-05: un fallo de carga a OCI, o una fidelidad bajo el umbral,
        no producen `error` — producen `exito_con_advertencias`. Pero el status
        no puede decir `exito` cuando algo falló: eso sería mentir en el contrato.
        """
        degradado = (
            self.almacenamiento_oci.status_upload is not StatusUpload.COMPLETADO
            or not self.evaluacion_calidad.supero_umbral
        )
        if degradado and self.status is StatusOperacion.EXITO:
            raise ValueError(
                "status='exito' es incoherente: la carga a OCI fallo o la fidelidad "
                "quedo bajo el umbral. Debe ser 'exito_con_advertencias' (decision D-05)."
            )
        return self

    # -------------------------------------------------------------------------

    def items_tipados(self) -> list[Any]:
        """Los ítems ya validados como objetos del modelo que corresponde."""
        modelo = MODELO_ITEM_POR_FORMATO[self.metadatos.formato_generado]
        return [modelo.model_validate(b) for b in self.contenido_adaptado.items]

    def nombre_objeto_oci(self) -> str:
        """
        Convención de nomenclatura (FR-PER-04), derivada del ejemplo oficial:
            contenido-vcn-principiante-flashcards-001.json
        """
        from .enums import normalizar

        slug = "-".join(normalizar(self.contenido_adaptado.titulo).split()[:4])
        perfil = "-".join(normalizar(self.metadatos.perfil_aplicado.value).split())
        formato = "-".join(normalizar(self.metadatos.formato_generado.value).split())
        return f"contenido-{slug}-{perfil}-{formato}.json"
