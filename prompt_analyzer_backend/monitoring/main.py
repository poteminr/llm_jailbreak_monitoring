import uvicorn
from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest

app = FastAPI()

REQUEST_COUNTER = Counter('ping_requests_total', 'Total number of ping requests')

@app.get('/ping')
async def ping():
    REQUEST_COUNTER.inc()
    return

@app.get('/metrics')
async def metrics():
    return Response(content=generate_latest(), media_type='text/plain')
