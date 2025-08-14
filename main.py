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
You will collect the following details from the user, one at a time, like filling in a form:
1. Name
2. Phone
3. Email
4. Service issue (what needs fixing)
5. Date (YYYY-MM-DD)
6. Start time (HH:MM)
7. End time (HH:MM)
8. Address
9. Latitude
10. Longitude

Rules:
- Ask only one missing field at a time.
- If the user provides multiple fields at once, accept them and then ask for the next missing field.
- Always confirm the collected value before moving on.
- Once all fields are collected, show the full summary in JSON and say: BOOK_READY <JSON>.
- Do NOT make up any data — wait for the user to provide it.
- Keep responses short and friendly.
"""


def all_fields_present(data):
    required = ["name","phone","email","issue","date","start","end","address","lat","lon"]
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

    # Token usage info
    token_info = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens
    }

    # Try to extract structured data if present in reply
    if "BOOK_READY" in reply:
        import json, re
        try:
            json_str = re.search(r"\{.*\}", reply, re.S).group()
            data = json.loads(json_str)
            sessions[sid]["data"].update(data)

            if all_fields_present(sessions[sid]["data"]):
                # Send booking request
                r = requests.post(API_BOOK_URL, json=sessions[sid]["data"])
                booking_resp = r.json()
                sessions.pop(sid)
                return {
                    "reply": f"✅ Booking confirmed! Your job ID is {booking_resp.get('jobId')}",
                    "tokens": token_info
                }
        except Exception as e:
            reply += f"\n\n(⚠ Error parsing booking info: {e})"

    return {"reply": reply, "tokens": token_info}