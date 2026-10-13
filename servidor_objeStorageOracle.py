"""
servidor_objeStorageOracle.py

Servidor MCP (FastMCP) que expone tools para interactuar con Oracle Object
Storage dentro del proyecto NuevaMente. Corre por stdio (subprocess local),
igual patrón que servidor_wiki.py del proyecto MCPMULTIAgentsResearch.

Autenticación: Instance Principal (sin claves en .env) — requiere que la
instancia esté en la Dynamic Group "nuevamente-instance-dg" con la Policy
correspondiente sobre el bucket configurado.

Aislamiento por usuario (app pública):
  Todas las tools reciben `user_id` (el `sub` de Google, lo pone la app, nunca
  el usuario ni un LLM). El servidor es la frontera de confianza: valida el
  user_id y arma las claves físicas del bucket.

  Identificadores LÓGICOS (los que ven el grafo, la UI y Chroma; no llevan el
  user_id):
      fuentes/<nombre>                      generados/formateados/<nombre>
  Claves FÍSICAS en el bucket (las arma este servidor):
      fuentes/<user_id>/<nombre>            generados/formateados/<user_id>/<nombre>
      generados/<user_id>/<objeto_id>       (guardar_resultado)

  Así ningún usuario puede listar, descargar ni pisar objetos de otro, aunque
  manipule el objeto_id. Los objetos que ya estaban en la raíz de fuentes/ y
  generados/ (antes del aislamiento) quedan como legado: nadie los ve.

Límites (configurables por variable de entorno):
  - OCI_MAX_DOCS_POR_USUARIO (default 5): documentos fuente (.pdf, .md, .txt) por usuario.
  - OCI_MAX_REPORTES_POR_USUARIO (default 20): reportes HTML guardados con el botón
    "Guardar en OCI" de la app. Se guardan en la misma carpeta del usuario, pero no
    se listan como documentos fuente, no se pueden usar como fuente y no cuentan
    para el límite de documentos.
  - OCI_MAX_UPLOAD_MB (default 10): tamaño máximo por archivo subido.

Tools expuestas (todas con user_id al final):
  - listar_documentos_fuente(user_id)
  - subir_documento_fuente(nombre_archivo, contenido_base64, user_id)
  - descargar_documento(objeto_id, user_id)
  - guardar_resultado(objeto_id, contenido, user_id)
  - guardar_resultado_formateado(nombre_archivo, contenido, user_id)
"""

import base64
import io
import os
import re
from datetime import datetime

import oci
from pypdf import PdfReader
from mcp.server.fastmcp import FastMCP as MCPServer

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

BUCKET_NAME = os.environ.get("OCI_BUCKET_NAME", "nuevamente-storage")
PREFIJO_FUENTES = "fuentes/"
PREFIJO_GENERADOS = "generados/"
PREFIJO_FORMATEADOS = "generados/formateados/"

EXTENSIONES_FUENTE = frozenset({".pdf", ".md", ".txt"})   # documentos que se pueden procesar
EXTENSIONES_REPORTE = frozenset({".html"})                # reportes que guarda la app
EXTENSIONES_PERMITIDAS = EXTENSIONES_FUENTE | EXTENSIONES_REPORTE


def _entero_env(nombre: str, default: int) -> int:
    try:
        valor = int(os.environ.get(nombre, default))
        return valor if valor > 0 else default
    except ValueError:
        return default


MAX_DOCS_POR_USUARIO = _entero_env("OCI_MAX_DOCS_POR_USUARIO", 5)
MAX_REPORTES_POR_USUARIO = _entero_env("OCI_MAX_REPORTES_POR_USUARIO", 20)
MAX_UPLOAD_BYTES = _entero_env("OCI_MAX_UPLOAD_MB", 10) * 1024 * 1024
MAX_PAGINAS_PDF = _entero_env("OCI_MAX_PAGINAS_PDF", 300)

server = MCPServer("nuevamente-object-storage")

# --------------------------------------------------------------------------
# Cliente OCI cacheado: se autentica y resuelve el namespace UNA sola vez
# por proceso, en vez de repetirlo en cada llamada a una tool.
# --------------------------------------------------------------------------

_client_cache = None
_namespace_cache = None

# Variables de entorno para autenticar con una API key (usuario IAM dedicado)
# fuera de OCI, p. ej. Streamlit Community Cloud, donde no hay Instance
# Principal ni ~/.oci/config. Con secrets de nivel superior de Streamlit llegan
# como variables de entorno, y el cliente MCP las hereda en el subproceso.
_VARS_API_KEY = (
    "OCI_USER_OCID",
    "OCI_TENANCY_OCID",
    "OCI_FINGERPRINT",
    "OCI_REGION",
    "OCI_PRIVATE_KEY",
)


def _config_desde_entorno() -> dict | None:
    """
    Arma la config de OCI a partir de variables de entorno (auth por API key).

    Devuelve None si NO hay ninguna de las variables (se sigue con Instance
    Principal / config file como siempre). Si hay algunas pero faltan otras,
    falla con un mensaje claro en vez de caer en silencio a otro método.
    El contenido de la clave nunca se imprime ni se incluye en errores.
    """
    valores = {nombre: (os.environ.get(nombre) or "").strip() for nombre in _VARS_API_KEY}
    if not any(valores.values()):
        return None

    faltan = [nombre for nombre, valor in valores.items() if not valor]
    if faltan:
        raise RuntimeError(
            "Configuración de OCI incompleta. Faltan variables: " + ", ".join(faltan)
        )

    clave = valores["OCI_PRIVATE_KEY"]
    # Algunos paneles de secrets guardan los saltos de línea como "\n" literales.
    if "\\n" in clave and "\n" not in clave:
        clave = clave.replace("\\n", "\n")

    config = {
        "user": valores["OCI_USER_OCID"],
        "tenancy": valores["OCI_TENANCY_OCID"],
        "fingerprint": valores["OCI_FINGERPRINT"],
        "region": valores["OCI_REGION"],
        "key_content": clave,
    }
    oci.config.validate_config(config)
    return config


def _get_client():
    """
    En la VM de producción usa Instance Principal automáticamente.
    En desarrollo local, seteá OCI_LOCAL_DEV=1 para saltar directo al
    config file y evitar la demora larga del timeout de Instance Principal.

    Fuera de OCI (p. ej. Streamlit Community Cloud) usa una API key de un
    usuario IAM dedicado si están definidas OCI_USER_OCID, OCI_TENANCY_OCID,
    OCI_FINGERPRINT, OCI_REGION y OCI_PRIVATE_KEY (ver _config_desde_entorno).

    Cachea el cliente y el namespace en memoria: la autenticación y el
    get_namespace() solo se hacen la primera vez que se llama dentro de
    este proceso; las siguientes llamadas devuelven el cache directo.
    """
    global _client_cache, _namespace_cache

    if _client_cache is not None:
        return _client_cache, _namespace_cache

    config_api_key = None if os.environ.get("OCI_LOCAL_DEV") == "1" else _config_desde_entorno()

    if os.environ.get("OCI_LOCAL_DEV") == "1":
        config = oci.config.from_file()
        client = oci.object_storage.ObjectStorageClient(config)
    elif config_api_key is not None:
        client = oci.object_storage.ObjectStorageClient(config_api_key)
    else:
        try:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
        except Exception:
            config = oci.config.from_file()
            client = oci.object_storage.ObjectStorageClient(config)

    namespace = client.get_namespace().data

    _client_cache = client
    _namespace_cache = namespace
    return client, namespace


# --------------------------------------------------------------------------
# Validaciones (frontera de confianza)
# --------------------------------------------------------------------------

_USER_ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CARACTERES_DE_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_LONGITUD_RUTA = 512


def _validar_user_id(user_id) -> str:
    """El user_id va como componente de la ruta física: solo [A-Za-z0-9_-], 1 a 64."""
    if not isinstance(user_id, str) or not _USER_ID_VALIDO.fullmatch(user_id):
        raise ValueError("user_id inválido.")
    return user_id


def _validar_ruta_relativa(ruta, *, permitir_subcarpetas: bool) -> str:
    """
    Valida un nombre/ruta relativa que se agrega DESPUÉS del prefijo del usuario.
    Rechaza vacío, '..', '.', segmentos vacíos ('//'), barras invertidas, '/' inicial
    y caracteres de control. Si no se permiten subcarpetas, tampoco acepta '/'.
    """
    if not isinstance(ruta, str) or not ruta or len(ruta) > _MAX_LONGITUD_RUTA:
        raise ValueError("Nombre de objeto inválido.")
    if _CARACTERES_DE_CONTROL.search(ruta) or "\\" in ruta or ruta.startswith("/"):
        raise ValueError("Nombre de objeto inválido.")
    segmentos = ruta.split("/")
    if not permitir_subcarpetas and len(segmentos) != 1:
        raise ValueError("Nombre de objeto inválido.")
    if any(s in ("", ".", "..") for s in segmentos):
        raise ValueError("Nombre de objeto inválido.")
    return ruta


def _extension(nombre: str) -> str:
    return ("." + nombre.rsplit(".", 1)[-1].lower()) if "." in nombre else ""


def _clave_fuente(user_id: str, nombre: str) -> str:
    return f"{PREFIJO_FUENTES}{_validar_user_id(user_id)}/{nombre}"


def _nombre_desde_id_logico_fuente(objeto_id) -> str:
    """'fuentes/manual.pdf' -> 'manual.pdf'. Exige el prefijo lógico fuentes/."""
    if not isinstance(objeto_id, str) or not objeto_id.startswith(PREFIJO_FUENTES):
        raise ValueError("Documento no autorizado.")
    try:
        return _validar_ruta_relativa(objeto_id[len(PREFIJO_FUENTES):], permitir_subcarpetas=True)
    except ValueError:
        raise ValueError("Documento no autorizado.") from None


def _extraer_texto(nombre_objeto: str, contenido_bytes: bytes) -> str:
    """
    Extrae texto plano de un documento según su extensión.
    PDF -> pypdf. Markdown/texto -> decodifica directo como UTF-8.
    """
    extension = nombre_objeto.rsplit(".", 1)[-1].lower() if "." in nombre_objeto else ""

    if extension == "pdf":
        lector = PdfReader(io.BytesIO(contenido_bytes))
        if len(lector.pages) > MAX_PAGINAS_PDF:
            raise ValueError(
                f"El PDF supera el máximo de {MAX_PAGINAS_PDF} páginas permitido."
            )
        paginas = [pagina.extract_text() or "" for pagina in lector.pages]
        return "\n\n".join(paginas).strip()

    # Markdown y texto plano se leen directo
    try:
        return contenido_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return contenido_bytes.decode("latin-1")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _listar_claves_usuario(client, namespace: str, user_id: str) -> list:
    """Objetos del usuario bajo fuentes/<user_id>/ (con paginación)."""
    prefijo = _clave_fuente(user_id, "")
    objetos = []
    inicio = None
    while True:
        kwargs = dict(
            namespace_name=namespace,
            bucket_name=BUCKET_NAME,
            prefix=prefijo,
            fields="size,timeModified",
        )
        if inicio:
            kwargs["start"] = inicio
        respuesta = client.list_objects(**kwargs)
        objetos.extend(respuesta.data.objects)
        inicio = getattr(respuesta.data, "next_start_with", None)
        if not inicio:
            return objetos


@server.tool()
def listar_documentos_fuente(user_id: str) -> list[dict]:
    """
    Lista los documentos del USUARIO bajo su prefijo (fuentes/<user_id>/) para
    que elija uno existente (Flujo A) en vez de subir un archivo nuevo.

    Args:
        user_id: identificador del usuario (lo pone la app).

    Devuelve una lista de dicts con: objeto_id (lógico, "fuentes/<nombre>"),
    nombre, tamano_bytes, fecha_modificacion.
    """
    client, namespace = _get_client()
    prefijo = _clave_fuente(user_id, "")

    documentos = []
    for objeto in _listar_claves_usuario(client, namespace, user_id):
        nombre = objeto.name[len(prefijo):]
        # Salteamos el "objeto" que representa la carpeta vacía en sí y los reportes HTML
        if not nombre or _extension(nombre) not in EXTENSIONES_FUENTE:
            continue
        documentos.append({
            "objeto_id": f"{PREFIJO_FUENTES}{nombre}",
            "nombre": nombre,
            "tamano_bytes": objeto.size,
            "fecha_modificacion": (
                objeto.time_modified.isoformat() if objeto.time_modified else None
            ),
        })

    return documentos


@server.tool()
def subir_documento_fuente(nombre_archivo: str, contenido_base64: str, user_id: str) -> dict:
    """
    Sube un documento nuevo (PDF, Markdown o texto plano) al prefijo del
    usuario (fuentes/<user_id>/). Usado en el Flujo B (subida desde la UI).

    Args:
        nombre_archivo: nombre original del archivo, ej. "manual_vcn.pdf"
            (sin carpetas; solo .pdf, .md o .txt).
        contenido_base64: contenido del archivo codificado en base64
            (así viaja de forma segura como texto a través de la tool MCP).
        user_id: identificador del usuario (lo pone la app).

    Límites: OCI_MAX_UPLOAD_MB por archivo y OCI_MAX_DOCS_POR_USUARIO documentos
    por usuario (reemplazar un archivo existente con el mismo nombre no suma).

    Devuelve: {objeto_id (lógico), bucket, status_upload}.
    """
    _validar_user_id(user_id)
    nombre = _validar_ruta_relativa(nombre_archivo, permitir_subcarpetas=False)
    extension = _extension(nombre)
    if extension not in EXTENSIONES_PERMITIDAS:
        raise ValueError(
            "Tipo de archivo no permitido. Usá: " + ", ".join(sorted(EXTENSIONES_FUENTE))
        )
    if not isinstance(contenido_base64, str):
        raise ValueError("Contenido inválido.")
    # Cota del tamaño decodificado ANTES de decodificar (base64 = 4 chars por cada 3 bytes).
    if (len(contenido_base64) * 3) // 4 > MAX_UPLOAD_BYTES + 3:
        raise ValueError(
            f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    try:
        contenido_bytes = base64.b64decode(contenido_base64, validate=True)
    except ValueError:
        raise ValueError("Contenido inválido.") from None
    if len(contenido_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    if not contenido_bytes:
        raise ValueError("El archivo está vacío.")

    client, namespace = _get_client()
    clave = _clave_fuente(user_id, nombre)

    existentes = {o.name for o in _listar_claves_usuario(client, namespace, user_id)}
    if clave not in existentes:
        if extension in EXTENSIONES_FUENTE:
            usados = sum(1 for n in existentes if _extension(n) in EXTENSIONES_FUENTE)
            if usados >= MAX_DOCS_POR_USUARIO:
                raise ValueError(
                    f"Alcanzaste el máximo de {MAX_DOCS_POR_USUARIO} documentos. "
                    "Reemplazá uno existente o usá uno de tu lista."
                )
        else:
            usados = sum(1 for n in existentes if _extension(n) in EXTENSIONES_REPORTE)
            if usados >= MAX_REPORTES_POR_USUARIO:
                raise ValueError(
                    f"Alcanzaste el máximo de {MAX_REPORTES_POR_USUARIO} reportes guardados."
                )

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=clave,
        put_object_body=contenido_bytes,
    )

    return {
        "objeto_id": f"{PREFIJO_FUENTES}{nombre}",
        "bucket": BUCKET_NAME,
        "status_upload": "completado",
    }


@server.tool()
def descargar_documento(objeto_id: str, user_id: str) -> dict:
    """
    Descarga un documento del usuario (prefijo fuentes/<user_id>/) y devuelve
    su texto ya extraído, listo para chunking. Usada por ambos flujos
    (A: documento elegido de la lista; B: recién subido).

    Args:
        objeto_id: identificador LÓGICO, ej. "fuentes/manual_vcn.pdf" (tal como
            lo devuelve listar_documentos_fuente o subir_documento_fuente).
        user_id: identificador del usuario (lo pone la app).

    Devuelve: {objeto_id, nombre, texto_extraido, tamano_bytes}.
    """
    _validar_user_id(user_id)
    nombre = _nombre_desde_id_logico_fuente(objeto_id)
    if _extension(nombre) not in EXTENSIONES_FUENTE:
        raise ValueError("Documento no autorizado.")  # los reportes HTML no son fuentes

    client, namespace = _get_client()
    respuesta = client.get_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=_clave_fuente(user_id, nombre),
    )
    contenido_bytes = respuesta.data.content

    texto_extraido = _extraer_texto(nombre, contenido_bytes)

    return {
        "objeto_id": objeto_id,
        "nombre": nombre,
        "texto_extraido": texto_extraido,
        "tamano_bytes": len(contenido_bytes),
    }


@server.tool()
def guardar_resultado(objeto_id: str, contenido: dict, user_id: str) -> dict:
    """
    Guarda el resultado generado (JSON del Redactor/Revisor/Modificador)
    bajo generados/<user_id>/ del bucket. Reservada exclusivamente para
    outputs del pipeline — nunca para documentos fuente.

    Args:
        objeto_id: nombre a usar dentro de la carpeta del usuario, ej.
            "contenido-vcn-principiante-flashcards-001.json". Cada
            modificación del Agente Modificador debe pasar un objeto_id
            nuevo (versión), nunca reusar uno existente.
        contenido: el dict serializable a JSON con el resultado completo.
        user_id: identificador del usuario (lo pone la app).

    Devuelve: {bucket, objeto_id (lógico: "generados/<objeto_id>"), status_upload}.
    """
    import json

    _validar_user_id(user_id)
    nombre = _validar_ruta_relativa(objeto_id, permitir_subcarpetas=True)

    client, namespace = _get_client()
    clave = f"{PREFIJO_GENERADOS}{user_id}/{nombre}"
    contenido_json = json.dumps(contenido, ensure_ascii=False, indent=2)

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=clave,
        put_object_body=contenido_json.encode("utf-8"),
    )

    return {
        "bucket": BUCKET_NAME,
        "objeto_id": f"{PREFIJO_GENERADOS}{nombre}",
        "status_upload": "completado",
    }


@server.tool()
def guardar_resultado_formateado(nombre_archivo: str, contenido: dict, user_id: str) -> dict:
    """
    Guarda el resultado generado por el pipeline (JSON del Redactor/Revisor/
    Modificador) bajo generados/formateados/<user_id>/, usando el nombre del
    archivo fuente original (versionado) como identificador.

    Agregada como función independiente de guardar_resultado() -- no la
    reemplaza; conviven las dos. El nodo Guardado del grafo (5to spec de
    NuevaMente) llama a esta.

    Args:
        nombre_archivo: nombre versionado, ej. "manual_vcn_v1.pdf.json". Cada
            modificación del Agente Modificador debe pasar un nombre nuevo
            (versión), nunca reusar uno existente.
        contenido: el dict serializable a JSON con el resultado completo.
        user_id: identificador del usuario (lo pone la app).

    Devuelve: {bucket, objeto_id (lógico: "generados/formateados/<nombre>"),
    status_upload} — literalmente el bloque almacenamiento_oci del esquema de salida.
    """
    import json

    _validar_user_id(user_id)
    nombre = _validar_ruta_relativa(nombre_archivo, permitir_subcarpetas=True)

    client, namespace = _get_client()
    clave = f"{PREFIJO_FORMATEADOS}{user_id}/{nombre}"
    contenido_json = json.dumps(contenido, ensure_ascii=False, indent=2)

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=clave,
        put_object_body=contenido_json.encode("utf-8"),
    )

    return {
        "bucket": BUCKET_NAME,
        "objeto_id": f"{PREFIJO_FORMATEADOS}{nombre}",
        "status_upload": "completado",
    }


# --------------------------------------------------------------------------
# Prueba manual aislada
# --------------------------------------------------------------------------

def _prueba_conexion():
    """
    Prueba manual end-to-end contra el bucket real: sube un documento de prueba,
    lo lista, lo descarga y verifica extracción de texto y aislamiento entre
    dos usuarios, y guarda resultados dummy.
    Correr con: python servidor_objeStorageOracle.py --test
    """
    print("=== Prueba de conexión a Object Storage ===")

    usuario_a = "prueba-usuario-a"
    usuario_b = "prueba-usuario-b"
    contenido_prueba = "Este es un documento de prueba para NuevaMente.\nSegunda línea."
    contenido_b64 = base64.b64encode(contenido_prueba.encode("utf-8")).decode("ascii")

    print("\n1. Subiendo documento de prueba (usuario A)...")
    resultado_subida = subir_documento_fuente("prueba_conexion.txt", contenido_b64, usuario_a)
    print("   ->", resultado_subida)

    print("\n2. Listando documentos fuente (usuario A)...")
    documentos = listar_documentos_fuente(usuario_a)
    print(f"   -> {len(documentos)} documento(s) encontrado(s)")
    for doc in documentos:
        print("     -", doc["nombre"])

    print("\n3. Descargando y extrayendo texto (usuario A)...")
    resultado_descarga = descargar_documento(resultado_subida["objeto_id"], usuario_a)
    print("   -> texto extraído:", repr(resultado_descarga["texto_extraido"]))

    assert resultado_descarga["texto_extraido"].strip() == contenido_prueba.strip(), (
        "El texto extraído no coincide con el original"
    )
    print("   -> OK, coincide con el original")

    print("\n4. Aislamiento: el usuario B no ve ni descarga lo del usuario A...")
    assert not any(d["nombre"] == "prueba_conexion.txt" for d in listar_documentos_fuente(usuario_b))
    try:
        descargar_documento(resultado_subida["objeto_id"], usuario_b)
    except Exception as exc:
        print("   -> OK, B no pudo descargar:", type(exc).__name__)
    else:
        raise AssertionError("El usuario B pudo descargar un documento del usuario A")

    print("\n5. Guardando resultado de prueba en generados/<user_id>/...")
    resultado_guardado = guardar_resultado(
        "prueba-resultado.json",
        {"status": "exito", "nota": "esto es una prueba"},
        usuario_a,
    )
    print("   ->", resultado_guardado)

    print("\n6. Guardando resultado de prueba en generados/formateados/<user_id>/...")
    resultado_guardado_formateado = guardar_resultado_formateado(
        "prueba-resultado-formateado.json",
        {"status": "exito", "nota": "esto es una prueba del 5to spec"},
        usuario_a,
    )
    print("   ->", resultado_guardado_formateado)

    print("\n=== Prueba completa OK ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _prueba_conexion()
    else:
        server.run()