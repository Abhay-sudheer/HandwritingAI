from __future__ import annotations

import os
import platform
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .services.images import prepare_image
from .services.recognition import Recognizer

BASE = Path(__file__).resolve().parents[2]
DB = Path(os.getenv("DATABASE_PATH", str(BASE / "backend" / "data" / "history.db")))
if not DB.is_absolute(): DB = (BASE / DB).resolve()
recognizer: Recognizer | None = None

def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB); db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, filename TEXT, text TEXT, created_at TEXT, duration_ms REAL, status TEXT, lines INTEGER, provider TEXT)")
    return db

@asynccontextmanager
async def lifespan(app):
    global recognizer
    recognizer = Recognizer()
    connect().close()
    yield

app = FastAPI(title="InkSight AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    r = recognizer
    return {"status":"ok", "python_version":platform.python_version(), "model_available":bool(r and r.ready), "model_loaded":bool(r and r.ready), "provider":r.provider if r else os.getenv("INFERENCE_PROVIDER","local"), "device":r.device if r and r.device else None, "message":r.error if r else "Starting"}

@app.get("/api/model-status")
def model_status():
    r=recognizer
    return {"available":bool(r and r.ready),"loaded":bool(r and r.ready),"provider":r.provider if r else "local","device":r.device if r else None,"model_path":str(r.model_path) if r and r.provider=="local" else None,"message":r.error if r else "Starting"}

@app.get("/api/stats")
def stats():
    with connect() as db:
        row=db.execute("SELECT COUNT(*) total, COALESCE(SUM(status='success'),0) successes, COALESCE(AVG(CASE WHEN status='success' THEN duration_ms END),0) average FROM history").fetchone()
    return {"total":row["total"],"successful":row["successes"],"average_duration_ms":round(row["average"],1),"model_ready":bool(recognizer and recognizer.ready)}

@app.post("/api/recognize")
async def recognize(file: UploadFile=File(...), segmentation_mode: str|None=Form(None)):
    allowed={"image/jpeg","image/png"}
    if file.content_type not in allowed: raise HTTPException(415,"Please upload a JPG or PNG image.")
    segmentation_mode=segmentation_mode or os.getenv("SEGMENTATION_MODE","auto")
    if segmentation_mode not in {"auto","single"}: raise HTTPException(422,"segmentation_mode must be 'auto' or 'single'.")
    max_bytes=int(os.getenv("MAX_UPLOAD_MB","15"))*1024*1024
    data=await file.read(max_bytes+1)
    if len(data)>max_bytes: raise HTTPException(413,f"Image exceeds the {max_bytes//1024//1024} MB upload limit.")
    try: prepared=prepare_image(data, segmentation_mode, int(os.getenv("MAX_IMAGE_DIMENSION","4000")))
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    r=recognizer
    if not r or not r.ready:
        detail=(r.error if r else None) or "Recognition model is not ready. Check backend diagnostics."
        raise HTTPException(503,detail)
    start=time.perf_counter()
    try:
        recognized=[r.recognize_line(line) for line in prepared.lines]
        text="".join(("\n\n" if i in prepared.paragraph_breaks else "\n") + value if i else value for i, value in enumerate(recognized) if value)
    except Exception as exc:
        raise HTTPException(502,f"Recognition failed using {r.provider}: {type(exc).__name__}. Check backend logs.") from exc
    duration=round((time.perf_counter()-start)*1000,1)
    record={"id":str(uuid.uuid4()),"filename":Path(file.filename or "upload").name,"text":text,"created_at":datetime.now(timezone.utc).isoformat(),"duration_ms":duration,"status":"success" if text else "empty","lines":len(prepared.lines),"provider":r.provider,"diagnostics":prepared.diagnostics}
    with connect() as db: db.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?,?)",tuple(record[k] for k in ("id","filename","text","created_at","duration_ms","status","lines","provider")))
    return record

@app.get("/api/history")
def history():
    with connect() as db: rows=db.execute("SELECT * FROM history ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]

@app.delete("/api/history/{record_id}")
def delete_history(record_id:str):
    with connect() as db: cur=db.execute("DELETE FROM history WHERE id=?",(record_id,))
    if not cur.rowcount: raise HTTPException(404,"History record not found.")
    return {"deleted":True}

@app.delete("/api/history")
def clear_history():
    with connect() as db: db.execute("DELETE FROM history")
    return {"deleted":True}
