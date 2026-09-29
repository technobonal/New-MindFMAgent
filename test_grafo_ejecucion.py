import asyncio
from grafo import construir_grafo
from agent_state import AgentState


async def _test():
    print(">>> Construyendo grafo...")
    grafo = await construir_grafo()
    print(">>> Grafo construido, invocando...")

    estado_inicial: AgentState = {
        "mensajes": [],
        "thread_id": "prueba-001",
        "tema_pedido_chat": "prueba_conexion",
        "candidatos_documento": [],
        "objeto_id_confirmado": None,
        "ejecucion_confirmada": False,
        "intentos_redactor": 0,
        "vueltas_modificacion": 0,
        "status": "",
    }

    config = {"configurable": {"thread_id": "prueba-001"}}

    print(">>> Llamando a grafo.ainvoke()... (esto puede tardar unos segundos, no cortes)")
    resultado = await grafo.ainvoke(estado_inicial, config=config)
    print(">>> ainvoke() terminó")

    print("=== Estado tras la primera pausa ===")
    print(resultado)

    await grafo.checkpointer.conn.close()
    print(">>> Listo.")


asyncio.run(_test())