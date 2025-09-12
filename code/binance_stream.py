import asyncio
import json
import websockets
import pandas as pd

symbols = ["BNBUSDT"] 
stream_names = "/".join(f"{s.lower()}@aggTrade" for s in symbols)

WS_COMBINED = f"wss://stream.binance.com:9443/stream?streams={stream_names}"

async def handle_combined(msg, df):
    data = msg.get("data", {})
    data = {k: data[k] for k in ["a", "p", "q", "f", "l", "T", "m", "M"]}
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    return df

async def run():
    df = pd.DataFrame(columns=["a","p","q","f","l","T","m","M"])
    async with websockets.connect(WS_COMBINED, ping_interval=20) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            df = await handle_combined(msg, df)

if __name__ == "__main__":
    asyncio.run(run())
