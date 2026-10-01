"""
rag/vectorstore.py

Chunking + indexado + búsqueda en Chroma para NuevaMente.

Diseño:
  - UNA sola colección ("documentos") en espacio COSENO.
  - Aislamiento por metadata: cada chunk lleva user_id y objeto_id, y toda
    consulta/borrado filtra por ambos. Dos usuarios (o dos documentos) nunca
    se mezclan.
  - Ingesta idempotente: si (user_id, objeto_id) ya está indexado con el mismo
    contenido (hash), no reindexa; si el contenido cambió, reemplaza.
  - Devuelve SIMILITUD coseno (1 - distancia). El umbral (0.78) NO se aplica
    acá: lo aplica el Investigador.

Firmas que espera grafo.py:
  indexar_documento(texto, objeto_id, user_id) -> None
  buscar_chunks_con_id(consulta, k, user_id, objeto_id) -> [(id, texto, similitud), ...] desc
      (el Redactor necesita los IDs de chunk para el campo `anclaje` de cada item)
  buscar_chunks(consulta, k, user_id, objeto_id) -> [(texto, similitud), ...] desc
      (se conserva tal cual; ya no la usa el grafo)

Extra para el Revisor:
  calcular_anclaje(afirmaciones, user_id, objeto_id) -> dict
"""

from __future__ import annotations

import hashlib
import os
import threading

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import obtener_embeddings

DIRECTORIO_CHROMA = os.environ.get("CHROMA_DIR", "chroma_db")
NOMBRE_COLECCION = "documentos"

# e5-base admite 512 tokens; ~1000 caracteres (~250 tokens) deja margen.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

_store: Chroma | None = None
_lock = threading.Lock()


def _obtener_store() -> Chroma:
    global _store
    with _lock:
        if _store is None:
            _store = Chroma(
                collection_name=NOMBRE_COLECCION,
                embedding_function=obtener_embeddings(),
                persist_directory=DIRECTORIO_CHROMA,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return _store


def _filtro(user_id: str, objeto_id: str) -> dict:
    return {"$and": [{"user_id": {"$eq": user_id}}, {"objeto_id": {"$eq": objeto_id}}]}


def indexar_documento(texto: str, objeto_id: str, user_id: str) -> None:
    texto = (texto or "").strip()
    if not texto:
        raise ValueError(f"El documento '{objeto_id}' no tiene texto extraíble (¿PDF escaneado?).")

    # El tamaño de chunk entra en el hash: si cambia CHUNK_SIZE/CHUNK_OVERLAP,
    # los documentos ya indexados se reindexan solos.
    hash_contenido = hashlib.sha256(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{texto}".encode("utf-8")).hexdigest()
    store = _obtener_store()
    coleccion = store._collection
    filtro = _filtro(user_id, objeto_id)

    existentes = coleccion.get(where=filtro, limit=1, include=["metadatas"])
    if existentes["ids"]:
        if existentes["metadatas"][0].get("hash_contenido") == hash_contenido:
            return  # ya indexado con el mismo contenido
        coleccion.delete(where=filtro)  # el documento cambió: reemplazar

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(texto)
    if not chunks:
        raise ValueError(f"No se generaron chunks para '{objeto_id}'.")

    id_base = hashlib.sha256(f"{user_id}|{objeto_id}".encode("utf-8")).hexdigest()[:16]
    store.add_texts(
        texts=chunks,
        metadatas=[
            {
                "user_id": user_id,
                "objeto_id": objeto_id,
                "hash_contenido": hash_contenido,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ],
        ids=[f"{id_base}-{i}" for i in range(len(chunks))],
    )


def buscar_chunks(consulta: str, k: int, user_id: str, objeto_id: str) -> list[tuple[str, float]]:
    resultados = _obtener_store().similarity_search_with_score(
        consulta, k=k, filter=_filtro(user_id, objeto_id)
    )
    # Espacio coseno: distancia = 1 - similitud. Ya vienen ordenados de mejor a peor.
    return [(doc.page_content, 1.0 - distancia) for doc, distancia in resultados]


def buscar_chunks_con_id(
    consulta: str, k: int, user_id: str, objeto_id: str
) -> list[tuple[str, str, float]]:
    """
    Igual que buscar_chunks, pero devuelve también el ID del chunk en Chroma
    (el mismo '<hash>-<n>' con el que se indexó): [(id, texto, similitud), ...].

    Los IDs son estables entre corridas mientras el documento no cambie, así
    que el `anclaje` de cada item generado se puede rastrear hasta su chunk.
    """
    vector = obtener_embeddings().embed_query(consulta)
    res = _obtener_store()._collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=_filtro(user_id, objeto_id),
        include=["documents", "distances"],
    )
    ids = res["ids"][0]
    textos = res["documents"][0]
    distancias = res["distances"][0]
    return [
        (chunk_id, texto, max(0.0, 1.0 - distancia))
        for chunk_id, texto, distancia in zip(ids, textos, distancias)
    ]


def calcular_anclaje(afirmaciones: list[str], user_id: str, objeto_id: str) -> dict:
    """
    anclaje_fuente_score: para cada afirmación del contenido generado, la
    similitud coseno con su chunk fuente más cercano; el score es el promedio.
    Se devuelve también el mínimo y el detalle por afirmación, porque un
    promedio alto puede tapar una afirmación sin respaldo (posible alucinación).
    """
    afirmaciones = [a.strip() for a in afirmaciones if a and a.strip()]
    if not afirmaciones:
        return {"score": None, "minimo": None, "por_afirmacion": []}

    vectores = obtener_embeddings().embed_queries(afirmaciones)
    res = _obtener_store()._collection.query(
        query_embeddings=vectores,
        n_results=1,
        where=_filtro(user_id, objeto_id),
        include=["distances"],
    )
    similitudes = [
        max(0.0, 1.0 - distancias[0]) if distancias else 0.0
        for distancias in res["distances"]
    ]
    return {
        "score": round(sum(similitudes) / len(similitudes), 4),
        "minimo": round(min(similitudes), 4),
        "por_afirmacion": [
            {"afirmacion": a, "similitud": round(s, 4)} for a, s in zip(afirmaciones, similitudes)
        ],
    }


if __name__ == "__main__":
    # Prueba de humo contra HF real + Chroma local: python -m rag.vectorstore
    usuario, doc = "prueba@local", "fuentes/prueba.txt"
    texto = (
        "Oracle Object Storage guarda objetos dentro de buckets. "
        "La capa Always Free incluye 20 GB de almacenamiento. "
        "Instance Principal permite autenticar una instancia OCI sin claves en el .env. "
    ) * 5

    indexar_documento(texto, doc, usuario)
    indexar_documento(texto, doc, usuario)  # 2da vez: debe ser no-op

    for consulta in ("¿Cómo se autentica una instancia sin claves?", "receta de paella valenciana"):
        print(f"\n{consulta}")
        for chunk, sim in buscar_chunks(consulta, 2, usuario, doc):
            print(f"  {sim:.3f}  {chunk[:70]}...")
        for chunk_id, chunk, sim in buscar_chunks_con_id(consulta, 2, usuario, doc):
            print(f"  [{chunk_id}] {sim:.3f}  {chunk[:50]}...")

    print("\nanclaje:", calcular_anclaje(["Instance Principal evita guardar claves"], usuario, doc))
    _obtener_store()._collection.delete(where=_filtro(usuario, doc))