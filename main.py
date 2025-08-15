# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from dotenv import load_dotenv
import boto3, os, uuid, json, re, requests
from openai import OpenAI
from graph import build_booking_graph, BookingState

load_dotenv()

# ----- ENV / Clients -----
DDB_TABLE = os.getenv("DDB_TABLE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BOOK_URL = os.environ.get("API_BOOK_URL", "http://localhost:8000/book")

if not DDB_TABLE:
    raise RuntimeError("DDB_TABLE is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

table = boto3.resource("dynamodb").Table(DDB_TABLE)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----- Compile LangGraph -----
booking_graph = build_booking_graph(table, client)

# ----- FastAPI App -----
app = FastAPI()

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")

# --- helper to make DynamoDB Decimals JSON-safe ---
def to_jsonable(o: Any):
    if isinstance(o, list):
        return [to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, Decimal):
        return float(o)
    return o

# ====== Schemas ======
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

class ChatRequest(BaseModel):
    jobId: str
    question: str

# ====== Endpoints ======
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

def explain_status_text(job: Dict[str, Any], tech: Optional[Dict[str, Any]],
                        tone: str = "friendly", locale: str = "en", max_words: int = 90) -> str:
    ctx_json = json.dumps(to_jsonable({"job": job, "technician": tech}))
    prompt = f"""
You are ServiceAI. Write a {tone} message in {locale} for the customer, based ONLY on this JSON:

{ctx_json}

Guidelines:
- Greet the customer by name.
- State current status and the appointment date with time window (no timezone changes).
- Mention assigned technician name if available and what they'll do.
- Give a next step or expectation (e.g., we'll text when the tech is on the way).
- Keep it under {max_words} words.
- Return PLAIN TEXT only (no JSON, no markdown).
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Safe fallback if LLM call fails
        slot = job.get("slot", {})
        tech_name = tech.get("name") if tech else None
        return (
            f"Hi {job.get('customer',{}).get('name','there')}, your job {job.get('id')} is "
            f"{job.get('status','PENDING')} for {slot.get('date','(date TBD)')} "
            f"{slot.get('start','?')}–{slot.get('end','?')}. "
            f"{('Your technician is ' + tech_name + '. ') if tech_name else ''}"
            f"We’ll keep you posted."
        )

@app.get("/status/{job_id}", response_class=HTMLResponse)
def get_status(job_id: str, format: str = "html"):
    resp = table.get_item(Key={"id": job_id})
    if "Item" not in resp:
        explanation = f"Job {job_id} not found."
        return HTMLResponse(f"<p>{explanation}</p>") if format == "html" else PlainTextResponse(explanation)

    job = resp["Item"]
    tech = None
    if job.get("assignedTechId"):
        t = table.get_item(Key={"id": job["assignedTechId"]})
        tech = t.get("Item")

    job_json = to_jsonable(job)
    tech_json = to_jsonable(tech) if tech else None

    prompt = f"""
    You are a service assistant. 
    Provide a short, friendly update about the customer's current booking, including the job status, date/time, location, and assigned technician name. 
    Avoid unnecessary details unless requested.

    Job:
    {json.dumps(job_json)}

    Technician:
    {json.dumps(tech_json)}
    """
    resp_llm = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    explanation = resp_llm.choices[0].message.content.strip()

    if format == "text":
        return PlainTextResponse(explanation)
    return HTMLResponse(f"<p>{explanation}</p>")


@app.post("/chat")
def chat(req: ChatRequest):
    j = table.get_item(Key={"id": req.jobId}).get("Item")
    if not j:
        return {"error": "Job not found", "jobId": req.jobId}
    tech = None
    if j.get("assignedTechId"):
        tech = table.get_item(Key={"id": j["assignedTechId"]}).get("Item")

    context = {"job": j, "technician": tech}
    prompt = f"""
You are ServiceAI. Answer using ONLY this JSON context and be concise (<80 words).
If asked for ETA, use the scheduled window; if uncertain, state the window and offer to notify.

CONTEXT:
{json.dumps(to_jsonable(context))}

USER QUESTION:
{req.question}
"""
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

# ====== Chatbot LLM (form-filling) ======
sessions: Dict[str, Dict[str, Any]] = {}

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
{"name":"...", "phone":"...", "email":"...", "issue":"...", "date":"YYYY-MM-DD", "start":"HH:MM", "end":"HH:MM", "address":"...", "lat":"...", "lon":"..."}

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
        sessions[sid] = {"history": [{"role": "system", "content": SYSTEM_PROMPT}], "data": {}}

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
            m = re.search(r"\{.*?\}", reply, re.S)
            if not m:
                raise ValueError("No JSON object found after BOOK_READY")

            data = json.loads(m.group())

            # Normalize common variants
            if "service_issue" in data and "issue" not in data:
                data["issue"] = data["service_issue"]
            if "latitude" in data and "lat" not in data:
                data["lat"] = data["latitude"]
            if "longitude" in data and "lon" not in data:
                data["lon"] = data["longitude"]

            sessions[sid]["data"].update(data)

            if all_fields_present(sessions[sid]["data"]):
                b = sessions[sid]["data"]

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

                # POST to /book
                url = (API_BOOK_URL or "http://127.0.0.1:8000/book").rstrip("/")
                r = requests.post(url, json=payload, timeout=10)

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
                    "reply": f"Booking confirmed! Job ID: {job_id} — Assigned to {tech_id}",
                    "tokens": token_info
                }
            else:
                # Keep chatting for missing fields
                return {"reply": reply, "tokens": token_info}

        except Exception as http_err:
            print(f"[chatbot_llm] POST /book failed: {http_err}")
            reply += f"\n\n(⚠ Booking failed: {http_err})"
            return {"reply": reply, "tokens": token_info}

    # Not ready yet
    return {"reply": reply, "tokens": token_info}
