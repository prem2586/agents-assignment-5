"""
FastAPI server for the Financial Approval System.

Follows the official CopilotKit + LangGraph FastAPI quickstart pattern:
https://docs.copilotkit.ai/langgraph/quickstart?agent=bring-your-own
"""

import os
from contextlib import asynccontextmanager

from numpy import rint
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from backend.agent.checkpointer import create_async_checkpointer
from backend.config import LANGSMITH_API_KEY, LANGSMITH_PROJECT
from pydantic import BaseModel
from fastapi import HTTPException
from langgraph.types import Command

# Enable LangSmith tracing if configured
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

class ReviewDecision(BaseModel):
    approved: bool
    comments: str = ""


class ParallelApprovalRequest(BaseModel):
    thread_id: str
    manager: ReviewDecision
    finance: ReviewDecision


def build_graph(checkpointer):
    """Try the student's graph first; fall back to the demo graph."""
    try:
        from backend.agent.graph import create_approval_graph
        graph = create_approval_graph(checkpointer=checkpointer)
        print("[server] Loaded student approval graph")
    except Exception as exc:
        from backend.agent.demo_graph import create_demo_graph
        graph = create_demo_graph(checkpointer=checkpointer)
        print(f"[server] Student graph not ready ({type(exc).__name__}), using demo graph")
    return graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # AG-UI calls the graph's async API, so the checkpointer must be the async
    # AsyncSqliteSaver, which has to be opened inside the running event loop.
    async with create_async_checkpointer() as checkpointer:
        routes_before = list(app.router.routes)
        # Attach the LangGraph agent as an AG-UI endpoint (official CopilotKit pattern)
        app.state.approval_graph = build_graph(checkpointer)
        add_langgraph_fastapi_endpoint(
            app=app,
            agent=LangGraphAGUIAgent(
                name="approval_agent",
                description="Financial approval workflow agent",
                graph=app.state.approval_graph,
            ),
            path="/",
        )
        try:
            yield
        finally:
            # Drop the endpoint bound to this checkpointer before it closes, so a
            # second startup in the same process (e.g. tests) registers a fresh one.
            app.router.routes[:] = routes_before


# FastAPI app
app = FastAPI(title="Financial Approval System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "financial-approval-system"}

from fastapi import HTTPException

@app.get("/debug/interrupts/{thread_id}")
async def debug_interrupts(thread_id: str):
    graph = getattr(app.state, "approval_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Approval graph not initialized",
        )

    snapshot = await graph.aget_state({
        "configurable": {"thread_id": thread_id}
    })

    return {
        "thread_id": thread_id,
        "checkpoint_exists": snapshot.created_at is not None,
        "pending_nodes": list(snapshot.next),
        "interrupts": [
            {
                "node": task.name,
                "id": intr.id,
                "value": intr.value,
            }
            for task in snapshot.tasks
            for intr in task.interrupts
        ],
    }

@app.post("/parallel-approval")
async def parallel_approval(request: ParallelApprovalRequest):
    graph = getattr(app.state, "approval_graph", None)

    if graph is None:
        raise HTTPException(503, "Approval graph not initialized")

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    snapshot = await graph.aget_state(config)

    print("\n========== PARALLEL APPROVAL DEBUG ==========")
    print("Thread ID:", request.thread_id)
    print("Pending nodes:", snapshot.next)
    print("Checkpoint status:", snapshot.values.get("status"))

    for task in snapshot.tasks:
        print("Task name:", task.name)
        print("Interrupt count:", len(task.interrupts))

        for intr in task.interrupts:
            print("Interrupt ID:", intr.id)
            print("Interrupt value:", intr.value)

    print("============================================\n")

    print("\n========== CHECKPOINT DEBUG ==========")
    print("Requested thread:", request.thread_id)
    print("Snapshot config:", snapshot.config)
    print("Created at:", snapshot.created_at)
    print("Metadata:", snapshot.metadata)
    print("Pending nodes:", snapshot.next)
    print("Tasks:", snapshot.tasks)
    print("======================================\n")

    print("\n========== SAVED CHECKPOINTS ==========")

    checkpointer = graph.checkpointer
    count = 0

    async for checkpoint in checkpointer.alist(None):
        saved_config = checkpoint.config.get("configurable", {})

        print(
            "Saved thread:",
            saved_config.get("thread_id"),
            "| Checkpoint:",
            saved_config.get("checkpoint_id"),
        )

        count += 1

        if count >= 20:
            break

    print("=======================================\n")

    interrupt_ids = {}

    for task in snapshot.tasks:
        for intr in task.interrupts:
            if task.name == "parallel_manager_review":
                interrupt_ids["manager"] = intr.id
            elif task.name == "parallel_finance_review":
                interrupt_ids["finance"] = intr.id

    if len(interrupt_ids) != 2:
        raise HTTPException(
        status_code=409,
        detail={
            "message": "Both parallel reviews must be pending",
            "thread_id": request.thread_id,
            "pending_nodes": list(snapshot.next),
            "found_interrupts": list(interrupt_ids.keys()),
        },
    )

    resume_values = {
        interrupt_ids["manager"]: request.manager.model_dump(),
        interrupt_ids["finance"]: request.finance.model_dump(),
    }

    try:
        result = await graph.ainvoke(
            Command(resume=resume_values),
            config=config,
        )
        print("Manager approved:", result.get("manager_approved"))
        print("Finance approved:", result.get("finance_approved"))
        print("Status:", result.get("status"))
        print("Decisions:", result.get("decisions"))
    except Exception as exc:
        # Log the exception server-side; don't expose internals to clients.
        print(f"[parallel-approval] Resume failed: {exc}", flush=True)
        raise HTTPException(500, "Parallel approval failed")

    return {
        "request_id": result.get("request_id"),
        "status": result.get("status"),
        "current_stage": result.get("current_stage"),
        "final_approved": result.get("final_approved"),
        "manager_approved": result.get("manager_approved"),
        "finance_approved": result.get("finance_approved"),
        "decisions": result.get("decisions", []),
    }

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
