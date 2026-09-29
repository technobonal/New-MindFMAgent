import asyncio
from grafo import construir_grafo

async def _test():
    grafo = await construir_grafo()
    print("✅ Grafo compilado OK")
    await grafo.checkpointer.conn.close()

asyncio.run(_test())