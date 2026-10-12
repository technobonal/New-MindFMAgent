"""
agentes/agente_ilustrador.py

Ilustrador: asigna un ícono de la lista permitida (ICONOS_FLASHCARD) a CADA
flashcard, sin importar cuántas sean.

  1. Si el Redactor eligió un ícono válido, se respeta.
  2. Si vino vacío (null) o con un nombre fuera de la lista, se elige uno de
     forma determinística (sin LLM ni red) por coincidencia de palabras entre el
     texto de la card y una tabla de palabras clave por ícono.
  3. Si ninguna palabra coincide, se usa un ícono neutro.

En los pasos 2 y 3 se prefieren los íconos menos usados en la tanda, para que
no se repitan mientras haya otros disponibles. Si hay más cards que íconos, se
repiten, pero ninguna card queda sin ícono.

La clave de imagenes_items es la posición de la card en items (desde 1), igual
que antes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from agent_state import AgentState
from src.contracts.formatos import ICONOS_FLASHCARD

log = logging.getLogger(__name__)

ICONIFY = "https://api.iconify.design"
PREFIJO_ICONOS = "lucide"
COLOR_ICONO = "%234FC3F7"  # cian del tema ('#' codificado para la URL)

_PERMITIDOS = frozenset(ICONOS_FLASHCARD)

# Orden de preferencia cuando no hay ninguna coincidencia de palabras (íconos de
# propósito general primero), seguido del resto de la lista permitida.
_NEUTROS = (
    "lightbulb", "book-open", "puzzle", "layers", "flag", "target", "search",
    "eye", "list-checks", "workflow", "file-text", "message-square",
)
_ORDEN = tuple(n for n in _NEUTROS if n in _PERMITIDOS) + tuple(
    n for n in ICONOS_FLASHCARD if n not in _NEUTROS
)

# Palabras clave por ícono (español e inglés, sin tildes, en minúsculas).
# Las de 4 letras o menos deben coincidir exactas; las más largas coinciden por
# prefijo ("participa" cubre participan, participante, participación...).
_PALABRAS_CLAVE: dict[str, tuple[str, ...]] = {
    "users": ("equipo", "grupo", "persona", "participa", "rol", "roles", "stakeholder", "interesado",
              "miembro", "cliente", "team", "people", "member", "colaborador"),
    "user": ("usuario", "individuo", "responsable", "propietario", "owner", "perfil",
             "desarrollador", "developer", "master", "persona"),
    "calendar": ("calendario", "fecha", "cronograma", "agenda", "planifica", "planning", "sprint", "reunion",
                 "evento", "semanal", "mensual", "diario", "daily", "schedule", "meeting", "ceremonia"),
    "clock": ("hora", "horas", "duracion", "minuto", "tiempo", "time", "timebox", "plazo", "limite"),
    "timer": ("cronometro", "temporizador", "timer", "countdown"),
    "hourglass": ("espera", "pendiente", "demora", "retraso", "deadline", "vencimiento"),
    "target": ("objetivo", "meta", "proposito", "finalidad", "enfoque", "goal", "target", "aim"),
    "flag": ("hito", "milestone", "bandera", "entrega", "incremento", "terminado", "done"),
    "list-checks": ("checklist", "verificacion", "criterio", "aceptacion", "requisito", "validar", "completar"),
    "clipboard-list": ("backlog", "inventario", "tarea", "historia", "story", "pendientes", "plan", "lista"),
    "refresh-cw": ("iteracion", "ciclo", "retroaliment", "retrospectiva", "mejora", "adaptacion",
                   "iterativo", "feedback", "actualiza", "inspeccion"),
    "repeat": ("repetir", "repeticion", "recurrente", "rutina", "periodico", "loop", "bucle"),
    "workflow": ("flujo", "proceso", "pipeline", "etapa", "fase", "secuencia", "procedimiento", "workflow"),
    "git-branch": ("descompon", "desglos", "dividir", "dividen", "subtarea", "dependencia", "bifurca",
                   "rama", "alternativa", "fragmenta"),
    "message-square": ("comunicacion", "conversacion", "mensaje", "comentario", "discusion", "dialogo",
                       "comunica"),
    "megaphone": ("anuncio", "anuncia", "difunde", "presenta", "demo", "demostracion", "divulga"),
    "handshake": ("acuerdo", "colaboracion", "compromiso", "negociacion", "alianza", "contrato",
                  "consenso", "pacto", "coopera"),
    "eye": ("visibilidad", "transparencia", "observa", "monitor", "supervisa", "revision", "vision", "visible"),
    "search": ("buscar", "investiga", "analisis", "analiza", "explora", "descubre", "identifica",
               "diagnostico", "research"),
    "lightbulb": ("idea", "concepto", "innovacion", "creativo", "insight", "aprendizaje", "sugerencia",
                  "solucion", "principio"),
    "puzzle": ("componente", "pieza", "integracion", "encaja", "estructura", "modulo", "parte", "partes"),
    "layers": ("capa", "capas", "nivel", "niveles", "jerarquia", "apilado", "composicion"),
    "rocket": ("lanzamiento", "despliegue", "deploy", "release", "arranque", "mvp", "impulso", "producto"),
    "trending-up": ("crecimiento", "aumento", "progreso", "avance", "productividad", "burndown", "escalar"),
    "gauge": ("metrica", "medicion", "velocidad", "indicador", "kpi", "capacidad", "estimacion",
              "puntos", "throughput", "rendimiento"),
    "settings": ("configuracion", "ajuste", "parametro", "configura", "ajusta", "opcion"),
    "wrench": ("herramienta", "reparar", "mantenimiento", "soporte", "correccion", "tool"),
    "file-text": ("documento", "informe", "reporte", "especificacion", "documentacion", "acta",
                  "artefacto", "registro", "texto"),
    "book-open": ("guia", "manual", "definicion", "glosario", "aprende", "lectura", "estudio", "teoria", "libro"),
    "server": ("servidor", "instancia", "host", "hosting", "infraestructura"),
    "database": ("datos", "almacenamiento", "repositorio", "tabla", "sql", "storage", "bucket", "persistencia"),
    "cloud": ("nube", "cloud", "oracle", "oci", "aws", "azure", "saas", "iaas"),
    "network": ("red", "redes", "vcn", "subred", "conectividad", "topologia", "conexion", "gateway",
                "enrutamiento", "dns"),
    "cpu": ("procesador", "computo", "cpu", "hardware", "shape", "ocpu", "memoria"),
    "code": ("codigo", "programacion", "software", "script", "api", "programa", "implementacion", "comando"),
    "shield": ("seguridad", "proteccion", "proteger", "firewall", "riesgo", "cumplimiento", "defensa", "politica"),
    "lock": ("cifrado", "privacidad", "restringido", "bloqueo", "confidencial", "encriptacion", "autenticacion"),
    "key": ("clave", "credencial", "llave", "token", "autorizacion", "certificado", "permiso"),
    "globe": ("global", "region", "internet", "mundial", "publico", "dominio", "web", "internacional"),
}
# Solo se usan íconos que sigan en la lista permitida del contrato.
_PALABRAS_CLAVE = {n: p for n, p in _PALABRAS_CLAVE.items() if n in _PERMITIDOS}


def _url_icono(nombre: str) -> str:
    return f"{ICONIFY}/{PREFIJO_ICONOS}:{nombre}.svg?color={COLOR_ICONO}&height=96"


def _tokens(texto: str) -> list[str]:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return [t for t in re.split(r"[^a-z0-9]+", sin_tildes) if t]


def _coincide(token: str, palabra: str) -> bool:
    return token == palabra if len(palabra) <= 4 else token.startswith(palabra)


_CAMPOS_TEXTO = ("frente", "dorso", "pista_didactica", "imagen_descripcion")


def _palabras_del_tema(flashcards: list[tuple[int, dict]]) -> frozenset[str]:
    """
    Palabras que aparecen en la mayoría de las cards (p. ej. "sprint" en un documento
    sobre Sprint Planning): describen el tema del documento, no a cada card, así que
    no deben decidir el ícono. Solo se aplica con 3 o más cards.
    """
    total = len(flashcards)
    if total < 3:
        return frozenset()
    umbral = max(3, -(-total * 6 // 10))  # 60 % de las cards, redondeado hacia arriba
    frecuencia: Counter = Counter()
    for _, it in flashcards:
        palabras = set()
        for campo in _CAMPOS_TEXTO:
            palabras.update(_tokens(str(it.get(campo) or "")))
        frecuencia.update(palabras)
    return frozenset(p for p, n in frecuencia.items() if n >= umbral)


def _puntajes(item: dict, ignorar: frozenset[str] = frozenset()) -> dict[str, int]:
    """Puntaje por ícono según las palabras de la card. El frente pesa el triple."""
    tokens = _tokens(str(item.get("frente") or "")) * 3
    for campo in _CAMPOS_TEXTO[1:]:
        tokens += _tokens(str(item.get(campo) or ""))
    tokens = [t for t in tokens if t not in ignorar]
    puntajes: dict[str, int] = {}
    for nombre, palabras in _PALABRAS_CLAVE.items():
        total = sum(1 for t in tokens for p in palabras if _coincide(t, p))
        if total:
            puntajes[nombre] = total
    return puntajes


_PENALIZACION_REPETIDO = 3  # lo que vale una palabra del frente: repetir un ícono "cuesta" una coincidencia


def _elegir_icono(item: dict, uso: Counter, ignorar: frozenset[str] = frozenset()) -> str:
    """
    Ícono de respaldo: el de mejor puntaje, descontando lo ya usado en la tanda. Un
    ícono muy relevante puede repetirse; uno de coincidencia débil cede ante otro
    sin usar. Sin ninguna coincidencia, se toma el menos usado (neutros primero).
    """
    puntajes = _puntajes(item, ignorar)
    return min(
        _ORDEN,
        key=lambda n: (-(puntajes.get(n, 0) - _PENALIZACION_REPETIDO * uso[n]), uso[n], _ORDEN.index(n)),
    )


def asignar_iconos(items: list) -> dict[str, str]:
    """
    {posición (desde 1): nombre de ícono} para cada flashcard (item con "frente").
    Respeta el ícono del modelo si es válido; completa el resto con el respaldo.
    """
    flashcards = [
        (i, it) for i, it in enumerate(items, start=1) if isinstance(it, dict) and it.get("frente")
    ]
    asignados: dict[int, str] = {}
    uso: Counter = Counter()

    # 1) los íconos válidos que eligió el modelo
    for i, it in flashcards:
        nombre = str(it.get("icono_busqueda") or "").strip().lower()
        if nombre in _PERMITIDOS:
            asignados[i] = nombre
            uso[nombre] += 1
        elif nombre:
            log.info("ilustrador: icono fuera de la lista descartado: %r", nombre)

    # 2) respaldo para el resto, en orden
    if _ORDEN:
        ignorar = _palabras_del_tema(flashcards)
        for i, it in flashcards:
            if i not in asignados:
                nombre = _elegir_icono(it, uso, ignorar)
                asignados[i] = nombre
                uso[nombre] += 1

    por_respaldo = len(flashcards) - sum(
        1 for i, it in flashcards
        if str(it.get("icono_busqueda") or "").strip().lower() in _PERMITIDOS
    )
    if flashcards:
        log.info(
            "ilustrador: %d flashcard(s), %d con ícono del modelo, %d asignado(s) por respaldo",
            len(flashcards), len(flashcards) - por_respaldo, por_respaldo,
        )
    return {str(i): n for i, n in asignados.items()}


async def nodo_ilustrador(state: AgentState) -> dict:
    items = (state.get("contenido_adaptado") or {}).get("items") or []
    imagenes = {pos: _url_icono(nombre) for pos, nombre in asignar_iconos(items).items()}
    return {"imagenes_items": imagenes}