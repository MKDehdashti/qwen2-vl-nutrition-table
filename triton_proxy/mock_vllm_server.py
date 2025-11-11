from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: Request):
    data = await req.json()
    text = "Mock reply: Nutrition table detected at (x=20, y=50, w=400, h=250)."
    return {"choices": [{"message": {"content": text}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
