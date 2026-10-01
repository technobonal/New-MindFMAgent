"""
NuevaMente — Contratos tipados.

Esta carpeta es LA FRONTERA del sistema. Mientras estos modelos estén
congelados, los tres carriles de trabajo pueden avanzar en paralelo sin
pisarse: cada uno programa contra el contrato, no contra el código del otro.

Se congela el 22 de septiembre. Cambiarlo después obliga a coordinar a las
tres personas, así que se cambia sólo con acuerdo explícito.
"""

from .enums import (
    FORMATOS_IMPLEMENTADOS_MVP,
    ClaridadPedagogica,
    FormatoSalida,
    NichoSector,
    NivelDetalle,
    PerfilDestinatario,
    StatusOperacion,
    StatusUpload,
    normalizar,
)
from .formatos import (
    MODELO_ITEM_POR_FORMATO,
    ItemBase,
    ItemBloqueGuion,
    ItemFlashcard,
    ItemPasoTutorial,
    ItemPreguntaQuiz,
    ItemPuntoResumen,
    OpcionQuiz,
    validar_pasos_consecutivos,
)
from .request import SolicitudAdaptacion
from .response import (
    AlmacenamientoOci,
    ContenidoAdaptado,
    EvaluacionCalidad,
    MetadatosAprendizaje,
    PaqueteEducativo,
)

__all__ = [
    # enums
    "PerfilDestinatario",
    "FormatoSalida",
    "NichoSector",
    "NivelDetalle",
    "StatusOperacion",
    "StatusUpload",
    "ClaridadPedagogica",
    "FORMATOS_IMPLEMENTADOS_MVP",
    "normalizar",
    # items
    "ItemBase",
    "ItemFlashcard",
    "ItemPasoTutorial",
    "ItemPuntoResumen",
    "ItemPreguntaQuiz",
    "ItemBloqueGuion",
    "OpcionQuiz",
    "MODELO_ITEM_POR_FORMATO",
    "validar_pasos_consecutivos",
    # entrada / salida
    "SolicitudAdaptacion",
    "PaqueteEducativo",
    "MetadatosAprendizaje",
    "ContenidoAdaptado",
    "EvaluacionCalidad",
    "AlmacenamientoOci",
]
