import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8001/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            for _ in range(3):
                data = await websocket.recv()
                packet = json.loads(data)
                print(f"📡 Received packet: {packet['timestamp']} - Analysis: {packet['analysis']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
