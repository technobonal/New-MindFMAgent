"""
NuevaMente — Esquemas de `contenido_adaptado.items[]`, uno por formato.

Resuelve la ambigüedad A-08, que era el vacío más grande del documento fuente:
se exigen 5 formatos pedagógicos (p.2) y sólo se documenta la estructura de uno
(Flashcards, p.5).

  · ItemFlashcard    — 🟢 documentado en la fuente (p.5). NO se modifica.
  · ItemPasoTutorial — 🟡 definido por el equipo
  · ItemPuntoResumen — 🟡 definido por el equipo
  · ItemPreguntaQuiz — 🟡 definido por el equipo
  · ItemBloqueGuion  — ⬜ definido, NO implementado en el MVP

Todos los ítems comparten el campo `anclaje` (§2.6 del análisis), que lista los
chunks que sustentan la afirmación. Es lo que le da trabajo hecho al verificador
de fidelidad y lo que permite señalar en pantalla de dónde salió cada dato.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from .enums import FormatoSalida

TextoNoVacio = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


# =============================================================================
# Base común
# =============================================================================


class ItemBase(BaseModel):
    """
    Base de todo ítem generado.

    `anclaje` lista los identificadores de chunk que sustentan este ítem.
    Puede venir vacío —el modelo no siempre lo produce— pero un paquete con
    muchos ítems sin anclaje es, en sí mismo, una señal de alucinación.
    """

    model_config = ConfigDict(extra="forbid")

    anclaje: list[str] = Field(
        default_factory=list,
        description="IDs de los chunks del documento que sustentan este item.",
    )


# =============================================================================
# 1. Flashcards  —  🟢 ESQUEMA OFICIAL (p.5). No tocar.
# =============================================================================


class ItemFlashcard(ItemBase):
    frente: TextoNoVacio = Field(description="La pregunta o concepto, en el frente.")
    dorso: TextoNoVacio = Field(description="La respuesta o explicacion, en el dorso.")
    pista_didactica: TextoNoVacio = Field(
        description="Analogia o ayuda de memoria que facilita la retencion."
    )


# =============================================================================
# 2. Tutorial  —  🟡 definido por el equipo
# =============================================================================


class ItemPasoTutorial(ItemBase):
    """
    Criterio de diseño: `resultado_esperado` es lo que separa un tutorial de
    una lista de instrucciones. Permite que quien estudia se autoverifique
    sin ayuda de nadie.
    """

    paso_numero: int = Field(ge=1, description="Consecutivo, empezando en 1.")
    titulo_paso: TextoNoVacio = Field(description="Titulo corto y accionable.")
    instruccion: TextoNoVacio = Field(description="Que debe hacer quien estudia.")
    resultado_esperado: TextoNoVacio = Field(
        description="Como sabe que el paso le salio bien."
    )
    advertencia: str | None = Field(
        default=None, description="Error comun o riesgo propio de este paso."
    )


# =============================================================================
# 3. Resumen Ejecutivo (TL;DR)  —  🟡 definido por el equipo
# =============================================================================


class ItemPuntoResumen(ItemBase):
    """
    Criterio de diseño: el destinatario natural es `Gestor Ejecutivo`. Un
    resumen que sólo comprime el texto técnico no le sirve. Los tres campos
    obligan a la cadena  hecho -> consecuencia -> valor,  que es como decide
    ese perfil.
    """

    punto_clave: TextoNoVacio = Field(description="El hecho tecnico, en una frase.")
    implicacion: TextoNoVacio = Field(description="Que significa en la practica.")
    relevancia_negocio: TextoNoVacio = Field(
        description="Por que le importa a quien decide presupuesto."
    )


# =============================================================================
# 4. Quiz  —  🟡 definido por el equipo
# =============================================================================


class OpcionQuiz(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TextoNoVacio = Field(description="Identificador corto: a, b, c, d.")
    texto: TextoNoVacio

    @field_validator("id")
    @classmethod
    def _id_en_minuscula(cls, v: str) -> str:
        return v.strip().lower()


class ItemPreguntaQuiz(ItemBase):
    """
    El nombre del formato en el documento es "Quiz Interactivo CON
    JUSTIFICACIONES" (p.2): por eso `justificacion` es obligatorio.

    `analisis_distractores` es lo que convierte el quiz en material de estudio
    en vez de un examen, y es la base directa del diferencial opcional de
    retroalimentacion en tiempo real (p.6).
    """

    pregunta: TextoNoVacio = Field(description="Una sola idea por pregunta.")
    opciones: list[OpcionQuiz] = Field(min_length=3, max_length=5)
    respuesta_correcta: TextoNoVacio = Field(
        description="Debe coincidir con el id de una de las opciones."
    )
    justificacion: TextoNoVacio = Field(description="Por que la respuesta es correcta.")
    analisis_distractores: str | None = Field(
        default=None, description="Por que fallan las otras opciones."
    )

    @field_validator("respuesta_correcta")
    @classmethod
    def _normalizar(cls, v: str) -> str:
        return v.strip().lower()

    @model_validator(mode="after")
    def _respuesta_existe_entre_opciones(self) -> "ItemPreguntaQuiz":
        """
        Validación NO NEGOCIABLE.

        Que `respuesta_correcta` apunte a una opción inexistente es el error
        más común en quizzes generados por un LLM. Se valida en el esquema:
        no se confía en el modelo.
        """
        ids = {o.id for o in self.opciones}
        if self.respuesta_correcta not in ids:
            raise ValueError(
                f"respuesta_correcta={self.respuesta_correcta!r} no existe entre "
                f"las opciones {sorted(ids)}."
            )
        if len(ids) != len(self.opciones):
            raise ValueError("Hay ids de opcion repetidos.")
        return self


# =============================================================================
# 5. Guion de Clase  —  ⬜ definido, NO implementado en el MVP
# =============================================================================


class ItemBloqueGuion(ItemBase):
    """Listo para usarse si el margen de horas lo permite (decisión D-01)."""

    bloque_numero: int = Field(ge=1)
    titulo_bloque: TextoNoVacio
    duracion_estimada_segundos: int = Field(gt=0)
    guion: TextoNoVacio = Field(description="Texto literal que dice el instructor.")
    apoyo_visual: TextoNoVacio = Field(description="Que se muestra en pantalla.")


# =============================================================================
# Registro formato -> modelo de ítem
# =============================================================================

ItemPaquete = Union[
    ItemFlashcard,
    ItemPasoTutorial,
    ItemPuntoResumen,
    ItemPreguntaQuiz,
    ItemBloqueGuion,
]

MODELO_ITEM_POR_FORMATO: dict[FormatoSalida, type[ItemBase]] = {
    FormatoSalida.FLASHCARDS: ItemFlashcard,
    FormatoSalida.TUTORIAL: ItemPasoTutorial,
    FormatoSalida.RESUMEN_EJECUTIVO: ItemPuntoResumen,
    FormatoSalida.QUIZ: ItemPreguntaQuiz,
    FormatoSalida.GUION_CLASE: ItemBloqueGuion,
}


def validar_pasos_consecutivos(items: list[ItemPasoTutorial]) -> None:
    """Un tutorial con pasos 1, 2, 5 está mal generado. Se rechaza."""
    numeros = [i.paso_numero for i in items]
    if numeros != list(range(1, len(numeros) + 1)):
        raise ValueError(
            f"Los pasos del tutorial deben ser consecutivos desde 1. Se recibio: {numeros}"
        )
