from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, TypedDict
from decimal import Decimal
import boto3, os, json, uuid
from datetime import datetime
from langgraph.graph import StateGraph, END
from openai import OpenAI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any
import json
from decimal import Decimal
from dotenv import load_dotenv
import os
import re
load_dotenv()

DDB_TABLE = os.getenv("DDB_TABLE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import requests
API_BOOK_URL = os.environ.get("API_BOOK_URL", "http://localhost:8000/book")

# ----- AWS Setup -----
table = boto3.resource("dynamodb").Table(os.environ["DDB_TABLE"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ----- LangGraph State -----
class BookingState(TypedDict):
    job: Dict[str, Any]
    assignedTechId: Optional[str]
    notifications: List[str]

# ----- Nodes -----
def create_job(state: BookingState) -> BookingState:
    table.put_item(Item=state["job"])
    return state

def llm_assign_tech(state: BookingState) -> BookingState:
    techs = table.scan(
        FilterExpression="begins_with(#id, :t)",
        ExpressionAttributeNames={"#id": "id"},
        ExpressionAttributeValues={":t": "TECH#"}
    )["Items"]

    prompt = f"""
    You are a scheduling AI for a service company.
    Job: {json.dumps(state['job'], default=str)}
    Technicians: {json.dumps(techs, default=str)}

    Rules:
    - Prefer matching skills.
    - If multiple match, choose closest by lat/lon.
    - If tied, choose lowest todayLoad.
    Return ONLY the technician's 'id'.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    tech_id = resp.choices[0].message.content.strip()
    state["assignedTechId"] = tech_id
    return state

def update_job(state: BookingState) -> BookingState:
    table.update_item(
        Key={"id": state["job"]["id"]},
        UpdateExpression="SET #s = :s, assignedTechId = :t, updatedAt = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "ASSIGNED",
            ":t": state["assignedTechId"],
            ":u": datetime.utcnow().isoformat()
        }
    )
    return state

# ----- Build LangGraph -----
builder = StateGraph(BookingState)
builder.add_node("create_job", create_job)
builder.add_node("llm_assign_tech", llm_assign_tech)
builder.add_node("update_job", update_job)

builder.set_entry_point("create_job")
builder.add_edge("create_job", "llm_assign_tech")
builder.add_edge("llm_assign_tech", "update_job")
builder.add_edge("update_job", END)

booking_graph = builder.compile()

# ----- FastAPI App -----
app = FastAPI()
@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")

# Request schema
class BookingRequest(BaseModel):
    name: str
    phone: str
    email: str
    issue: str
    date: str
    start: str
    end: str
    address: str
    lat: float
    lon: float

@app.post("/book")
def book_service(req: BookingRequest):
    job_id = f"JOB#{uuid.uuid4().hex[:8]}"
    job_item = {
        "id": job_id,
        "status": "PENDING",
        "customer": {"name": req.name, "phone": req.phone, "email": req.email},
        "issue": req.issue,
        "slot": {"date": req.date, "start": req.start, "end": req.end},
        "location": {
            "lat": Decimal(str(req.lat)),
            "lon": Decimal(str(req.lon)),
            "address": req.address
        },
        "assignedTechId": None,
        "createdAt": datetime.utcnow().isoformat(),
        "updatedAt": datetime.utcnow().isoformat()
    }

    initial_state: BookingState = {"job": job_item, "assignedTechId": None, "notifications": []}
    result = booking_graph.invoke(initial_state)

    return {
        "jobId": job_id,
        "status": "ASSIGNED",
        "assignedTechId": result["assignedTechId"]
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://end-to-end-service-mgmt.s3-website-ap-southeast-2.amazonaws.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helper to make DynamoDB Decimals JSON-safe ---
def to_jsonable(o: Any):
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, Decimal):
        return float(o)
    return o

# ========== STATUS LOOKUP ==========
@app.get("/status/{job_id}")
def get_status(job_id: str):
    resp = table.get_item(Key={"id": job_id})
    if "Item" not in resp:
        return {"error": "Job not found", "jobId": job_id}
    job = resp["Item"]
    tech = None
    if job.get("assignedTechId"):
        t = table.get_item(Key={"id": job["assignedTechId"]})
        tech = t.get("Item")
    return {
        "job": to_jsonable(job),
        "technician": to_jsonable(tech) if tech else None
    }

# ========== SIMPLE CHAT (LLM) ==========
class ChatRequest(BaseModel):
    jobId: str
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # fetch context
    j = table.get_item(Key={"id": req.jobId}).get("Item")
    if not j:
        return {"error": "Job not found", "jobId": req.jobId}
    tech = None
    if j.get("assignedTechId"):
        tech = table.get_item(Key={"id": j["assignedTechId"]}).get("Item")

    # build compact, grounded prompt
    context = {
        "job": j,
        "technician": tech,
    }
    prompt = f"""
You are ServiceAI. Answer using ONLY this JSON context and be concise (<80 words).
If asked for ETA, use the scheduled window; if uncertain, state the window and offer to notify.

CONTEXT:
{json.dumps(to_jsonable(context))}

USER QUESTION:
{req.question}
"""

    # call your LLM (same client you used for assignment)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()

    # (optional) store conversation
    table.put_item(Item={
        "id": f"MSG#{req.jobId}#{datetime.utcnow().isoformat()}Z",
        "jobId": req.jobId,
        "role": "user",
        "text": req.question,
    })
    table.put_item(Item={
        "id": f"MSG#{req.jobId}#{datetime.utcnow().isoformat()}Z_a",
        "jobId": req.jobId,
        "role": "assistant",
        "text": answer,
    })

    return {"answer": answer}

sessions = {}

class ChatMessage(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = """
You are a friendly automotive service booking assistant.

Collect these fields (exact keys) for the booking:
- name
- phone
- email
- issue
- date  (YYYY-MM-DD)
- start (HH:MM, 24h)
- end   (HH:MM, 24h)
- address
- lat   (number)
- lon   (number)

Rules:
- If the user provides ALL fields in one message, IMMEDIATELY output:

BOOK_READY
{"name":"...", "phone":"...", "email":"...", "issue":"...", "date":"YYYY-MM-DD", "start":"HH:MM", "end":"HH:MM", "address":"...", "lat":25.04, "lon":121.55}

- NO extra text before BOOK_READY.
- NO markdown/code fences.
- If some fields are missing, ask for exactly ONE missing field at a time, confirm it, and continue.
- Never invent values. Keep replies short.
"""

def all_fields_present(data: Dict[str, Any]) -> bool:
    required = ["name", "phone", "email", "issue", "date", "start", "end", "address", "lat", "lon"]
    return all(k in data and data[k] for k in required)

@app.post("/chatbot_llm")
def chatbot_llm(msg: ChatMessage):
    sid = msg.session_id
    if sid not in sessions:
        sessions[sid] = {
            "history": [{"role": "system", "content": SYSTEM_PROMPT}],
            "data": {}
        }

    sessions[sid]["history"].append({"role": "user", "content": msg.message})

    # Call LLM
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=sessions[sid]["history"]
    )
    reply = resp.choices[0].message.content
    sessions[sid]["history"].append({"role": "assistant", "content": reply})

    token_info = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens
    }

    # If booking complete
    if reply.strip().startswith("BOOK_READY"):
        try:
            # non-greedy JSON capture; guard if missing
            m = re.search(r"\{.*?\}", reply, re.S)
            if not m:
                raise ValueError("No JSON object found after BOOK_READY")

            data = json.loads(m.group())

            # Normalize common variants we've already seen
            if "service_issue" in data and "issue" not in data:
                data["issue"] = data["service_issue"]
            if "latitude" in data and "lat" not in data:
                data["lat"] = data["latitude"]
            if "longitude" in data and "lon" not in data:
                data["lon"] = data["longitude"]

            sessions[sid]["data"].update(data)

            if all_fields_present(sessions[sid]["data"]):
                b = sessions[sid]["data"]

                # Build payload exactly as /book expects
                payload = {
                    "name": b["name"],
                    "phone": b["phone"],
                    "email": b["email"],
                    "issue": b["issue"],
                    "date": b["date"],
                    "start": b["start"],
                    "end": b["end"],
                    "address": b["address"],
                    "lat": float(b["lat"]),
                    "lon": float(b["lon"])
                }

                # Single POST to /book
                url = (API_BOOK_URL or "http://127.0.0.1:8000/book").rstrip("/")
                r = requests.post(url, json=payload, timeout=10)

                # Capture body for debugging before raising
                raw_text = r.text
                status = r.status_code
                r.raise_for_status()

                try:
                    booking_resp = r.json()
                except ValueError:
                    print(f"[chatbot_llm] /book returned non-JSON (status {status}): {raw_text}")
                    raise

                job_id = booking_resp.get("jobId")
                tech_id = booking_resp.get("assignedTechId")
                if not job_id or not tech_id:
                    print(f"[chatbot_llm] /book JSON missing fields: {booking_resp}")
                    raise RuntimeError("Missing jobId or assignedTechId in /book response")

                sessions.pop(sid, None)
                return {
                    "reply": f"✅ Booking confirmed! Job ID: {job_id} — Assigned to {tech_id}",
                    "tokens": token_info
                }
            else:
                # Not all fields present; just continue the conversation
                return {"reply": reply, "tokens": token_info}

        except Exception as http_err:
            # Do NOT re-raise blindly; surface the error and keep the convo alive
            print(f"[chatbot_llm] POST /book failed: {http_err}")
            reply += f"\n\n(⚠ Booking failed: {http_err})"
            return {"reply": reply, "tokens": token_info}

    # Default: not ready yet, return the assistant message
    return {"reply": reply, "tokens": token_info}
