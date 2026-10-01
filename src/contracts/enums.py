"""
NuevaMente — Enumeraciones del contrato.

Resuelve las ambigüedades A-03, A-04 y A-06 del Decision Gate.

DECISIÓN SOBRE LA FORMA CANÓNICA
--------------------------------
El documento del Hackathon nombra los perfiles y formatos de dos maneras:

  · En la prosa (p.1-2), en forma larga:  "Principiante / Transición de Carrera"
  · En el Ejemplo de Solicitud (p.4), en forma corta:  "Principiante"

La forma CANÓNICA de este contrato es la CORTA, porque es la única que aparece
dentro de un contrato real en el documento. Así la respuesta que produce el
sistema coincide carácter por carácter con el ejemplo oficial de la p.5, que es
lo que un evaluador va a comparar.

La forma larga se acepta como alias en la entrada. La comparación ignora
mayúsculas, tildes y signos de puntuación, de modo que todas estas entradas
son válidas y equivalentes:

    "Principiante"
    "principiante"
    "Principiante / Transición de Carrera"
    "Principiante / Transicion de Carrera"
"""

from __future__ import annotations

import unicodedata
from enum import Enum


# =============================================================================
# Normalización para comparar alias
# =============================================================================


def normalizar(texto: str) -> str:
    """
    Reduce un texto a su forma comparable: sin tildes, sin mayúsculas,
    sin signos, sin espacios repetidos.

        "Guía Práctica Paso a Paso (Tutorial)"  ->  "guia practica paso a paso tutorial"
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    limpio = "".join(c if c.isalnum() else " " for c in sin_tildes.lower())
    return " ".join(limpio.split())


#: Registro de alias, FUERA del cuerpo de los Enum.
#: Declararlo dentro convertiria el atributo en un miembro del Enum y la clase
#: base dejaria de poder heredarse ("cannot extend enum").
_ALIAS_REGISTRY: dict[type, dict[str, str]] = {}


def registrar_alias(cls: type, alias: dict[str, str]) -> None:
    """Asocia {texto normalizado -> valor canonico} a una clase de enum."""
    _ALIAS_REGISTRY.setdefault(cls, {}).update(alias)


class _EnumConAlias(str, Enum):
    """Enum de texto que acepta alias, tildes y mayusculas al construirse."""

    @classmethod
    def _missing_(cls, value: object):
        if not isinstance(value, str):
            return None
        clave = normalizar(value)

        # 1. Alias declarado para esta clase
        canonico = _ALIAS_REGISTRY.get(cls, {}).get(clave)
        if canonico is not None:
            return cls._value2member_map_.get(canonico)

        # 2. Valor canonico, salvo tildes, mayusculas o puntuacion
        for miembro in cls:
            if normalizar(miembro.value) == clave:
                return miembro
        return None

    @classmethod
    def valores_aceptados(cls) -> list[str]:
        """Valores canonicos, para mensajes de error accionables."""
        return [m.value for m in cls]


# =============================================================================
# Perfil del destinatario  —  documento p.1
# =============================================================================


class PerfilDestinatario(_EnumConAlias):
    PRINCIPIANTE = "Principiante"
    DESARROLLADOR = "Desarrollador"
    LIDER_TECNICO = "Lider Tecnico"
    GESTOR_EJECUTIVO = "Gestor Ejecutivo"


registrar_alias(PerfilDestinatario, {
    normalizar("Principiante / Transición de Carrera"): "Principiante",
    normalizar("Transicion de Carrera"): "Principiante",
    normalizar("Desarrollador Junior / Semi Senior"): "Desarrollador",
    normalizar("Desarrollador Junior"): "Desarrollador",
    normalizar("Semi Senior"): "Desarrollador",
    normalizar("Líder Técnico / Arquitecto"): "Lider Tecnico",
    normalizar("Arquitecto"): "Lider Tecnico",
    normalizar("Gestor / Ejecutivo (No Técnico)"): "Gestor Ejecutivo",
    normalizar("Gestor"): "Gestor Ejecutivo",
    normalizar("Ejecutivo"): "Gestor Ejecutivo",
    normalizar("Ejecutivo No Tecnico"): "Gestor Ejecutivo",
})


# =============================================================================
# Formato pedagógico de salida  —  documento p.2
# =============================================================================


class FormatoSalida(_EnumConAlias):
    FLASHCARDS = "Flashcards"
    TUTORIAL = "Tutorial"
    QUIZ = "Quiz"
    RESUMEN_EJECUTIVO = "Resumen Ejecutivo"
    GUION_CLASE = "Guion de Clase"


registrar_alias(FormatoSalida, {
    normalizar("Flashcards de Memorización"): "Flashcards",
    normalizar("Guía Práctica Paso a Paso (Tutorial)"): "Tutorial",
    normalizar("Guia Practica Paso a Paso"): "Tutorial",
    normalizar("Guia Practica"): "Tutorial",
    normalizar("Quiz Interactivo con Justificaciones"): "Quiz",
    normalizar("Quiz Interactivo"): "Quiz",
    normalizar("Resumen Ejecutivo (TL;DR)"): "Resumen Ejecutivo",
    normalizar("TL;DR"): "Resumen Ejecutivo",
    normalizar("TLDR"): "Resumen Ejecutivo",
    normalizar("Guion de Clase / Video"): "Guion de Clase",
    normalizar("Guion de Video"): "Guion de Clase",
})


#: Decisión D-01: se implementan 4 de los 5 formatos.
#: El quinto queda declarado en el contrato y devuelve FORMATO_NO_DISPONIBLE_EN_MVP.
#: Esto es deliberado: demuestra que el contrato fue diseñado para los 5.
FORMATOS_IMPLEMENTADOS_MVP: frozenset[FormatoSalida] = frozenset(
    {
        FormatoSalida.FLASHCARDS,
        FormatoSalida.TUTORIAL,
        FormatoSalida.QUIZ,
        FormatoSalida.RESUMEN_EJECUTIVO,
    }
)


# =============================================================================
# Nicho / contexto de aplicación  —  documento p.2
# =============================================================================


class NichoSector(_EnumConAlias):
    FINTECH = "Fintech"
    SALUD = "Salud"
    ECOMMERCE = "E-commerce"
    GENERAL = "General"


registrar_alias(NichoSector, {
    normalizar("Ecommerce"): "E-commerce",
    normalizar("E commerce"): "E-commerce",
    normalizar("Comercio Electronico"): "E-commerce",
    normalizar("Health"): "Salud",
    normalizar("Salud y Bienestar"): "Salud",
})


# =============================================================================
# Nivel de detalle  —  ambigüedad A-06, resuelta por el equipo
# =============================================================================


class NivelDetalle(_EnumConAlias):
    """
    Modula EXTENSIÓN y DENSIDAD del contenido.

    Es ortogonal al perfil, que modula LENGUAJE y PROFUNDIDAD COGNITIVA.
    Un mismo perfil puede pedir más o menos detalle sin cambiar de registro.

    "Didactico" es el valor que aparece en el Ejemplo de Solicitud (p.4).
    Los otros dos son decisión del equipo: el documento no los define.
    """

    DIDACTICO = "Didactico"
    ESTANDAR = "Estandar"
    PROFUNDO = "Profundo"


registrar_alias(NivelDetalle, {
    normalizar("Didáctico"): "Didactico",
    normalizar("Estándar"): "Estandar",
    normalizar("Basico"): "Didactico",
    normalizar("Detallado"): "Profundo",
    normalizar("Avanzado"): "Profundo",
})


# =============================================================================
# Estados de la operación  —  decisión D-05 (degradación controlada)
# =============================================================================


class StatusOperacion(_EnumConAlias):
    #: Todo salió bien, incluida la carga a OCI. Valor del ejemplo oficial (p.4).
    EXITO = "exito"
    #: El paquete se generó, pero algo no crítico falló:
    #: carga a OCI fallida, o fidelidad bajo el umbral tras el reintento.
    EXITO_CON_ADVERTENCIAS = "exito_con_advertencias"
    #: No hay paquete que entregar.
    ERROR = "error"


class StatusUpload(_EnumConAlias):
    COMPLETADO = "completado"  # valor del ejemplo oficial (p.5)
    FALLIDO = "fallido"
    NO_INTENTADO = "no_intentado"


class ClaridadPedagogica(_EnumConAlias):
    """
    Escala de la evaluación pedagógica.

    El documento sólo evidencia el valor "Alta" (p.5). La escala completa
    es decisión del equipo y debe documentarse como tal en el README.
    """

    ALTA = "Alta"
    MEDIA = "Media"
    BAJA = "Baja"
    #: Se emite cuando el score de fidelidad quedó bajo el umbral
    #: tras el reintento (decisión D-03).
    REQUIERE_REVISION = "Requiere revision"


registrar_alias(NivelDetalle, {normalizar("Medio"): "Estandar"})
registrar_alias(ClaridadPedagogica, {
    normalizar("Requiere revisión"): "Requiere revision",
})
