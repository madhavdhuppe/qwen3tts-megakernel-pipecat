from fastapi import FastAPI

from server.routes import router

app = FastAPI(title="Qwen3-TTS Megakernel Pipecat Adapter")

app.include_router(router)
