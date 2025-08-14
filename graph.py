# graph.py
from typing import TypedDict, Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, END
import re
# Booking state shared between nodes
class BookingState(TypedDict):
    job: Dict[str, Any]
    assignedTechId: Optional[str]
    notifications: List[str]

def build_booking_graph(table, openai_client):
    """
    Build and return the compiled LangGraph for booking.
    `table` is a boto3 DynamoDB Table object.
    `openai_client` is an OpenAI() client.
    """
    # ----- Nodes (capture table & client via closure) -----
    def create_job(state: BookingState) -> BookingState:
        table.put_item(Item=state["job"])
        return state

    def llm_assign_tech(state: BookingState) -> BookingState:
        # Scan technicians (ids begin with TECH#)
        techs = table.scan(
            FilterExpression="begins_with(#id, :t)",
            ExpressionAttributeNames={"#id": "id"},
            ExpressionAttributeValues={":t": "TECH#"}
        )["Items"]

        prompt = f"""
        You are a scheduling AI for a service company.
        Job: {state['job']}
        Technicians: {techs}

        Rules:
        - Prefer matching skills.
        - If multiple match, choose closest by lat/lon.
        - If tied, choose lowest todayLoad.
        Return ONLY the technician's 'id'.
        """

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = resp.choices[0].message.content.strip()
        # ✅ Extract just TECH#<id> even if the model adds sentences/markdown
        m = re.search(r"\bTECH#\w+\b", raw)
        if not m:
            # optional: fail loud so you can see the exact LLM output in logs
            raise ValueError(f"Could not parse technician id from LLM output: {raw}")
        state["assignedTechId"] = m.group(0)
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

    return builder.compile()
