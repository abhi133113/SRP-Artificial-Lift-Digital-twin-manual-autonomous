import asyncio
import websockets
import json

async def trigger_websocket():
    uri = "ws://localhost:8005/websocket"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            # Keep connection open for 10 seconds to allow simulation to run
            for _ in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received data: {data['timestamp']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(trigger_websocket())
