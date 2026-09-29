"""
servidor_objeStorageOracle.py

Servidor MCP (FastMCP) que expone tools para interactuar con Oracle Object
Storage dentro del proyecto NuevaMente. Corre por stdio (subprocess local),
igual patrón que servidor_wiki.py del proyecto MCPMULTIAgentsResearch.

Autenticación: Instance Principal (sin claves en .env) — requiere que la
instancia esté en la Dynamic Group "nuevamente-instance-dg" con la Policy
correspondiente sobre el bucket configurado.

Organización del bucket por prefijos:
  - fuentes/               -> documentos que suben o eligen los usuarios (Flujo A/B)
  - generados/              -> outputs del Redactor/Revisor/Modificador (legacy)
  - generados/formateados/  -> outputs del pipeline, ruta fija usada por el
                                Buscador de Documentos (5to spec)

Tools expuestas:
  - listar_documentos_fuente()
  - subir_documento_fuente(objeto_id, contenido_base64)
  - descargar_documento(objeto_id)
  - guardar_resultado(objeto_id, contenido)
  - guardar_resultado_formateado(nombre_archivo, contenido)
"""

import base64
import io
import os
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

server = MCPServer("nuevamente-object-storage")

# --------------------------------------------------------------------------
# Cliente OCI cacheado: se autentica y resuelve el namespace UNA sola vez
# por proceso, en vez de repetirlo en cada llamada a una tool.
# --------------------------------------------------------------------------

_client_cache = None
_namespace_cache = None


def _get_client():
    """
    En la VM de producción usa Instance Principal automáticamente.
    En desarrollo local, seteá OCI_LOCAL_DEV=1 para saltar directo al
    config file y evitar la demora larga del timeout de Instance Principal.

    Cachea el cliente y el namespace en memoria: la autenticación y el
    get_namespace() solo se hacen la primera vez que se llama dentro de
    este proceso; las siguientes llamadas devuelven el cache directo.
    """
    global _client_cache, _namespace_cache

    if _client_cache is not None:
        return _client_cache, _namespace_cache

    if os.environ.get("OCI_LOCAL_DEV") == "1":
        config = oci.config.from_file()
        client = oci.object_storage.ObjectStorageClient(config)
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


def _extraer_texto(nombre_objeto: str, contenido_bytes: bytes) -> str:
    """
    Extrae texto plano de un documento según su extensión.
    PDF -> pypdf. Markdown/texto -> decodifica directo como UTF-8.
    """
    extension = nombre_objeto.rsplit(".", 1)[-1].lower() if "." in nombre_objeto else ""

    if extension == "pdf":
        lector = PdfReader(io.BytesIO(contenido_bytes))
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

@server.tool()
def listar_documentos_fuente() -> list[dict]:
    """
    Lista los documentos disponibles en el prefijo fuentes/ del bucket,
    para que el usuario elija uno existente (Flujo A) en vez de subir
    un archivo nuevo.

    Devuelve una lista de dicts con: objeto_id, nombre, tamano_bytes,
    fecha_modificacion.
    """
    client, namespace = _get_client()

    respuesta = client.list_objects(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        prefix=PREFIJO_FUENTES,
        fields="size,timeModified",
    )

    documentos = []
    for objeto in respuesta.data.objects:
        # Salteamos el "objeto" que representa la carpeta vacía en sí
        if objeto.name == PREFIJO_FUENTES:
            continue
        documentos.append({
            "objeto_id": objeto.name,
            "nombre": objeto.name.replace(PREFIJO_FUENTES, "", 1),
            "tamano_bytes": objeto.size,
            "fecha_modificacion": (
                objeto.time_modified.isoformat() if objeto.time_modified else None
            ),
        })

    return documentos


@server.tool()
def subir_documento_fuente(nombre_archivo: str, contenido_base64: str) -> dict:
    """
    Sube un documento nuevo (PDF, Markdown o texto plano) al prefijo
    fuentes/ del bucket. Usado en el Flujo B (subida directa desde la UI).

    Args:
        nombre_archivo: nombre original del archivo, ej. "manual_vcn.pdf".
        contenido_base64: contenido del archivo codificado en base64
            (así viaja de forma segura como texto a través de la tool MCP).

    Devuelve: {objeto_id, bucket, status_upload}.
    """
    client, namespace = _get_client()

    contenido_bytes = base64.b64decode(contenido_base64)
    objeto_id = f"{PREFIJO_FUENTES}{nombre_archivo}"

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=objeto_id,
        put_object_body=contenido_bytes,
    )

    return {
        "objeto_id": objeto_id,
        "bucket": BUCKET_NAME,
        "status_upload": "completado",
    }


@server.tool()
def descargar_documento(objeto_id: str) -> dict:
    """
    Descarga un documento del bucket (prefijo fuentes/) y devuelve su
    texto ya extraído, listo para chunking. Usada por ambos flujos
    (A: documento elegido de la lista; B: recién subido).

    Args:
        objeto_id: identificador completo del objeto, ej.
            "fuentes/manual_vcn.pdf" (tal como lo devuelve
            listar_documentos_fuente o subir_documento_fuente).

    Devuelve: {objeto_id, nombre, texto_extraido, tamano_bytes}.
    """
    client, namespace = _get_client()

    respuesta = client.get_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=objeto_id,
    )
    contenido_bytes = respuesta.data.content

    nombre = objeto_id.replace(PREFIJO_FUENTES, "", 1)
    texto_extraido = _extraer_texto(nombre, contenido_bytes)

    return {
        "objeto_id": objeto_id,
        "nombre": nombre,
        "texto_extraido": texto_extraido,
        "tamano_bytes": len(contenido_bytes),
    }


@server.tool()
def guardar_resultado(objeto_id: str, contenido: dict) -> dict:
    """
    Guarda el resultado generado (JSON del Redactor/Revisor/Modificador)
    en el prefijo generados/ del bucket. Reservada exclusivamente para
    outputs del pipeline — nunca para documentos fuente.

    Args:
        objeto_id: nombre a usar dentro de generados/, ej.
            "contenido-vcn-principiante-flashcards-001.json". Cada
            modificación del Agente Modificador debe pasar un objeto_id
            nuevo (versión), nunca reusar uno existente.
        contenido: el dict serializable a JSON con el resultado completo.

    Devuelve: {bucket, objeto_id, status_upload} — literalmente el bloque
    almacenamiento_oci del esquema de salida.
    """
    import json

    client, namespace = _get_client()

    objeto_id_completo = f"{PREFIJO_GENERADOS}{objeto_id}"
    contenido_json = json.dumps(contenido, ensure_ascii=False, indent=2)

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=objeto_id_completo,
        put_object_body=contenido_json.encode("utf-8"),
    )

    return {
        "bucket": BUCKET_NAME,
        "objeto_id": objeto_id_completo,
        "status_upload": "completado",
    }


@server.tool()
def guardar_resultado_formateado(nombre_archivo: str, contenido: dict) -> dict:
    """
    Guarda el resultado generado por el pipeline (JSON del Redactor/Revisor/
    Modificador) bajo la ruta fija generados/formateados/ del bucket,
    usando el nombre del archivo fuente original como identificador.

    Agregada como función independiente de guardar_resultado() -- no la
    reemplaza ni modifica su comportamiento; conviven las dos. El nodo
    Guardado del grafo (5to spec de NuevaMente) llama a esta.

    Args:
        nombre_archivo: nombre del archivo fuente original (sin prefijo),
            ej. "manual_vcn.pdf". Se usa tal cual dentro de
            generados/formateados/. Cada modificación del Agente
            Modificador debe pasar un nombre nuevo (versión), nunca
            reusar uno existente, mismo criterio que guardar_resultado().
        contenido: el dict serializable a JSON con el resultado completo.

    Devuelve: {bucket, objeto_id, status_upload} — literalmente el bloque
    almacenamiento_oci del esquema de salida.
    """
    import json

    client, namespace = _get_client()

    objeto_id_completo = f"{PREFIJO_GENERADOS}formateados/{nombre_archivo}"
    contenido_json = json.dumps(contenido, ensure_ascii=False, indent=2)

    client.put_object(
        namespace_name=namespace,
        bucket_name=BUCKET_NAME,
        object_name=objeto_id_completo,
        put_object_body=contenido_json.encode("utf-8"),
    )

    return {
        "bucket": BUCKET_NAME,
        "objeto_id": objeto_id_completo,
        "status_upload": "completado",
    }


# --------------------------------------------------------------------------
# Prueba manual aislada
# --------------------------------------------------------------------------

def _prueba_conexion():
    """
    Prueba manual end-to-end: sube un documento de prueba, lo lista,
    lo descarga y verifica extracción de texto, guarda un resultado
    dummy. Correr con: python servidor_objeStorageOracle.py
    """
    print("=== Prueba de conexión a Object Storage ===")

    contenido_prueba = "Este es un documento de prueba para NuevaMente.\nSegunda línea."
    contenido_b64 = base64.b64encode(contenido_prueba.encode("utf-8")).decode("ascii")

    print("\n1. Subiendo documento de prueba...")
    resultado_subida = subir_documento_fuente("prueba_conexion.txt", contenido_b64)
    print("   ->", resultado_subida)

    print("\n2. Listando documentos fuente...")
    documentos = listar_documentos_fuente()
    print(f"   -> {len(documentos)} documento(s) encontrado(s)")
    for doc in documentos:
        print("     -", doc["nombre"])

    print("\n3. Descargando y extrayendo texto...")
    resultado_descarga = descargar_documento(resultado_subida["objeto_id"])
    print("   -> texto extraído:", repr(resultado_descarga["texto_extraido"]))

    assert resultado_descarga["texto_extraido"].strip() == contenido_prueba.strip(), (
        "El texto extraído no coincide con el original"
    )
    print("   -> OK, coincide con el original")

    print("\n4. Guardando resultado de prueba en generados/...")
    resultado_guardado = guardar_resultado(
        "prueba-resultado.json",
        {"status": "exito", "nota": "esto es una prueba"},
    )
    print("   ->", resultado_guardado)

    print("\n5. Guardando resultado de prueba en generados/formateados/...")
    resultado_guardado_formateado = guardar_resultado_formateado(
        "prueba-resultado-formateado.json",
        {"status": "exito", "nota": "esto es una prueba del 5to spec"},
    )
    print("   ->", resultado_guardado_formateado)

    print("\n=== Prueba completa OK ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _prueba_conexion()
    else:
        server.run()