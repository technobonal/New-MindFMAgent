from rag.vectorstore import _obtener_store

col = _obtener_store()._collection
r = col.get(ids=["7e53b66f6dfb5248-37"])
print(r["documents"][0] if r["documents"] else "No existe ese ID")