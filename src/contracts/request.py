"""
NuevaMente — Contrato de entrada de la acción principal.

Implementa la decisión D-04: la acción principal acepta DOS MODOS.

  · Modo A — archivo:  el usuario carga un PDF, Markdown o texto.
                       La interfaz extrae el texto y llena `documento_contenido`.
  · Modo B — texto embebido:  `documento_titulo` + `documento_contenido`,
                       exactamente el Ejemplo de Solicitud oficial (p.4).

Internamente convergen: un solo núcleo, dos puertas. Así funciona tanto si un
evaluador carga un PDF como si toma el ejemplo de la p.4 y lo prueba tal cual.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .enums import (
    FORMATOS_IMPLEMENTADOS_MVP,
    FormatoSalida,
    NichoSector,
    NivelDetalle,
    PerfilDestinatario,
)

TextoNoVacio = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]

#: Por debajo de esto no hay material suficiente para chunking ni recuperación.
LONGITUD_MINIMA_CONTENIDO = 50


class SolicitudAdaptacion(BaseModel):
    """
    Contrato de entrada. Coincide campo por campo con el Ejemplo de Solicitud
    del documento oficial (p.4).

    Obligatorios (ambigüedad A-07, resuelta por el equipo):
        documento_titulo · documento_contenido · perfil_destinatario · formato_salida

    Con valor por defecto:
        nicho_sector   -> "General"
        nivel_detalle  -> "Estandar"
    """

    model_config = ConfigDict(
        extra="forbid",  # un campo no reconocido es un error, no se ignora en silencio
        use_enum_values=False,
        json_schema_extra={
            "example": {
                "documento_titulo": "Introduccion a la Arquitectura de Redes VCN en OCI",
                "documento_contenido": "La Virtual Cloud Network (VCN) es una red privada...",
                "perfil_destinatario": "Principiante",
                "formato_salida": "Flashcards",
                "nicho_sector": "General",
                "nivel_detalle": "Didactico",
            }
        },
    )

    documento_titulo: TextoNoVacio = Field(
        description="Titulo del material tecnico de origen.",
        max_length=300,
    )
    documento_contenido: TextoNoVacio = Field(
        description="Texto del documento. En modo archivo, lo llena la interfaz "
        "tras extraerlo del PDF, Markdown o texto plano."
    )
    perfil_destinatario: PerfilDestinatario = Field(
        description="A quien va dirigido. Acepta forma corta o larga."
    )
    formato_salida: FormatoSalida = Field(
        description="Formato pedagogico del paquete. Acepta forma corta o larga."
    )
    nicho_sector: NichoSector = Field(
        default=NichoSector.GENERAL,
        description="Sector que contextualiza ejemplos y analogias.",
    )
    nivel_detalle: NivelDetalle = Field(
        default=NivelDetalle.ESTANDAR,
        description="Extension y densidad. Ortogonal al perfil.",
    )

    # -------------------------------------------------------------------------

    @field_validator("documento_contenido")
    @classmethod
    def _contenido_suficiente(cls, v: str) -> str:
        if len(v.strip()) < LONGITUD_MINIMA_CONTENIDO:
            raise ValueError(
                f"El documento tiene menos de {LONGITUD_MINIMA_CONTENIDO} caracteres "
                f"utiles. No hay material suficiente para generar contenido educativo."
            )
        return v

    @field_validator("formato_salida")
    @classmethod
    def _formato_disponible(cls, v: FormatoSalida) -> FormatoSalida:
        """
        Decisión D-01: se implementan 4 de los 5 formatos. El quinto está
        declarado en el contrato pero no disponible en el MVP.
        """
        if v not in FORMATOS_IMPLEMENTADOS_MVP:
            disponibles = ", ".join(sorted(f.value for f in FORMATOS_IMPLEMENTADOS_MVP))
            raise ValueError(
                f"El formato '{v.value}' esta definido en el contrato pero no "
                f"implementado en el MVP. Formatos disponibles: {disponibles}."
            )
        return v

    # -------------------------------------------------------------------------

    def clave_cache_documento(self) -> str:
        """
        Identidad del documento para el índice vectorial (decisión D-07).

        Se deriva del CONTENIDO, no del título: el mismo documento cargado dos
        veces con nombres distintos reutiliza su índice en vez de reindexar.
        """
        import hashlib

        h = hashlib.sha256(self.documento_contenido.encode("utf-8")).hexdigest()
        return f"doc_{h[:16]}"
