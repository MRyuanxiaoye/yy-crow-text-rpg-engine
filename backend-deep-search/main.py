import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from app.graph import app as graph_app
from app.tools.explainer import TermExplainer
from pydantic import BaseModel

app = FastAPI(title="Deep Search Agent API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

explainer = TermExplainer()

class ChatRequest(BaseModel):
    query: str

class ExplainRequest(BaseModel):
    term: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/explain")
async def explain_term(request: ExplainRequest):
    """
    Lightweight drill-down for terms
    """
    definition = explainer.explain(request.term)
    return {"term": request.term, "definition": definition}

@app.get("/api/stream")
async def stream_chat(query: str):
    """
    SSE Endpoint for streaming graph events using astream_events for true token-level streaming
    """
    async def event_generator():
        seen_logs_global = set() # Global set for the entire stream session
        try:
            # Use astream_events to capture token-level events
            # version="v1" is required for LangGraph >= 0.2
            async for event in graph_app.astream_events({"query": query}, version="v1"):
                kind = event["event"]
                
                # 1. Stream Logs (from on_chain_end of specific nodes)
                # We look for chain ends that correspond to our graph nodes returning state updates
                if kind == "on_chain_end":
                    node_name = event.get("name")
                    output = event.get("data", {}).get("output")
                    
                    if output and isinstance(output, dict):
                        # Check for logs in the output state
                        if "logs" in output and output["logs"]:
                            print(f"LOG_CHAIN [Stream] Event from '{node_name}' | Logs in output: {len(output['logs'])}", flush=True)

                            # For simplicity, if we get a batch of logs, we yield them one by one
                            # (Deduplication logic might be needed if events are redundant, but usually nodes run once)
                            # seen_logs = set() # This was reset every event, causing duplicates across events!
                            for log in output["logs"]:
                                if log not in seen_logs_global:
                                    print(f"LOG_CHAIN [Stream] -> YIELDING: {log[:30]}...", flush=True)
                                    yield {
                                        "event": "log",
                                        "data": json.dumps({"content": log}, ensure_ascii=False)
                                    }
                                    seen_logs_global.add(log)
                                else:
                                    print(f"LOG_CHAIN [Stream] -> SKIPPING DUP: {log[:30]}...", flush=True)

                        # Check for final report completion (to extract keywords)
                        if "final_report" in output and node_name == "writer":
                            keywords = output.get("keywords", [])
                            # Send keywords event
                            yield {
                                "event": "report",
                                "data": json.dumps({
                                    "content": "",
                                    "full_content": True,
                                    "keywords": keywords
                                }, ensure_ascii=False)
                            }

                # 2. Stream Tokens (from Writer's LLM)
                # We look for chat model stream events specifically from the writer node
                elif kind == "on_chat_model_stream":
                    # Check if this event comes from the writer node
                    # In LangGraph, events usually have metadata['langgraph_node']
                    metadata = event.get("metadata", {})
                    if metadata.get("langgraph_node") == "writer":
                        content = event["data"]["chunk"].content
                        if content:
                            yield {
                                "event": "report",
                                "data": json.dumps({
                                    "content": content,
                                    "full_content": False,
                                    "keywords": []
                                }, ensure_ascii=False)
                            }

            # End of stream
            yield {
                "event": "end",
                "data": json.dumps({"status": "complete"})
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

