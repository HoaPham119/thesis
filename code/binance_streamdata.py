import asyncio
import json
import websockets
import pandas as pd

symbols = "BNBUSDT"
raw_cols = ["a", "p", "q", "f", "l", "T", "m", "M"]
stream_names = f"{symbols.lower()}@aggTrade"

WS_COMBINED = f"wss://stream.binance.com:9443/stream?streams={stream_names}"

async def handle_combined(msg, df):
    data = msg.get("data", {})
    data = {k: data[k] for k in raw_cols}
    print(data)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    return df

async def run():
    df = pd.DataFrame(columns=raw_cols)
    async with websockets.connect(WS_COMBINED, ping_interval=20) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            df = await handle_combined(msg, df)

if __name__ == "__main__":
    asyncio.run(run())
