"""
Node functions for the Financial Approval workflow.

Students implement 8 node functions and 6 routing functions that
form the LangGraph approval pipeline with risk-based escalation.

Part 1: LangGraph Workflow + Interrupts (35 points)
"""

import json

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from backend.agent.state import ApprovalState
from langchain_core.runnables import RunnableConfig
from backend.config import (
    get_llm,
    BUDGET_CEILING,
    DEPARTMENT_BUDGETS,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)
from backend.guardrails.input_validator import validate_request
from backend.guardrails.output_filter import sanitize_output
from backend.audit import log_audit_event
from backend.notifications import notify_reviewer


# ============================================================
# NODE FUNCTIONS (8 total)
# ============================================================

def submit_request(state: ApprovalState) -> dict:
    """
    Node 1: Receive and validate the financial request.
    

    TODO (3 points):
    - Extract request details from state
    - Call validate_request() to check the request
    - If invalid, set status to "rejected" and return with validation_message
    - If valid, set current_stage to "risk_assessment" and status to "pending"
    - Add an AIMessage summarizing the submitted request
    - Return updated state fields as a dict

    Hints:
    - validate_request() returns (is_valid: bool, message: str)
    - Return dict with keys: is_valid, validation_message, current_stage, status, messages
    - Also reset the previous request's review state ("decisions": [], risk_level /
      manager_approved / finance_approved / final_approved: None, ...): the frontend
      reuses one thread for every request in a chat session, so old values are still in
      the checkpoint (a request rejected here never runs assess_risk, so without the reset
      it would report the previous request's risk_level)
    - Reset department_budget / budget_remaining / within_budget to None, not 0: high and
      critical requests skip validate_budget, and the Approval Panel shows "Not checked"
      only when department_budget is None (0 shows "$0, Within budget")
    """

    title = state.get("title", "")
    amount = state.get("amount", 0)
    department = state.get("department", "")
    description = state.get("description", "")
    requester = state.get("requester", "")


    is_valid, validation_message = validate_request(
      amount=amount,
      department=department,
      title=title,
      description=description,
      justification=state.get("justification", "")
    )

    updates = {
        "is_valid": is_valid,
        "validation_message": validation_message,
        "decisions": [],
        "risk_level": None,
        "manager_approved": None,
        "finance_approved": None,
        "final_approved": None,
        "department_budget": None,
        "budget_remaining": None,
        "within_budget": None,
        "parallel_manager_decision": None,
        "parallel_finance_decision": None,
    }

    if not is_valid:
        updates.update({
            "status": "rejected",
            "current_stage": "validation",
            "messages": [
                AIMessage(
                    content=f"Request rejected: {validation_message}"
                )
            ],
        })
        return updates
    updates.update({
        "status": "pending",
        "current_stage": "risk_assessment",
        "messages": [
            AIMessage(
                content=(
                    f"Financial request submitted successfully.\n"
                    f"Title: {title}\n"
                    f"Amount: ${amount:,.2f}\n"
                    f"Department: {department}\n"
                    f"Requested by: {requester}\n"
                    f"Description: {description}\n"
                    f"Status: Pending risk assessment."
                )
            )
        ],
    })

    return updates


def assess_risk(state: ApprovalState) -> dict:
    """
    Node 2: Use LLM to assess risk level of the request.
    """

    # 1. Extract request details
    title = state.get("title", "")
    description = state.get("description", "")
    amount = state.get("amount", 0)
    department = state.get("department", "")
    justification = state.get("justification", "")
    priority = state.get("priority", "normal")

    # 2. Build risk assessment prompt
    prompt = f"""
    You are a financial risk assessment specialist.

    Evaluate the following financial request and classify its risk level.

    REQUEST DETAILS:
    Title: {title}
    Description: {description}
    Amount: ${amount}
    Department: {department}
    Justification: {justification}
    Priority: {priority}

    RISK CLASSIFICATION RULES:

    1. If priority is "urgent" AND amount > $50,000:
      risk_level = "critical"

    2. Otherwise, classify by amount:
      - Amount <= $10,000: low
      - Amount > $10,000 and <= $50,000: medium
      - Amount > $50,000 and <= $100,000: high
      - Amount > $100,000: critical

    Apply these rules in the specified order.
    Do not change the classification based on subjective factors.

    Return ONLY valid JSON in this format:

    {{
        "risk_level": "low",
        "risk_reasoning": "Brief explanation of classification"
    }}
    """

    # 3. Invoke LLM
    response = get_llm().invoke([
        HumanMessage(content=prompt)
    ])

    # Gemini and OpenAI return plain text through .text
    response_text = response.text.strip()

    # 4. Parse LLM response
    risk_level = "medium"
    risk_reasoning = "Unable to parse LLM risk assessment."

    try:
        # Remove optional Markdown JSON fences
        import re

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response_text,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(cleaned)

        parsed_level = str(
            result.get("risk_level", "")
        ).lower().strip()

        if parsed_level in {
            "low", "medium", "high", "critical"
        }:
            risk_level = parsed_level
            risk_reasoning = str(
                result.get(
                    "risk_reasoning",
                    "Risk classified based on request amount and priority."
                )
            )

    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback to medium when parsing fails
        pass

    # 5. Add risk assessment message
    message = (
        f"Risk Assessment Completed\n"
        f"Risk Level: {risk_level.upper()}\n"
        f"Reasoning: {risk_reasoning}"
    )

    # 6. Return updated state
    return {
        "risk_level": risk_level,
        "risk_reasoning": risk_reasoning,
        "current_stage": "risk_assessment",
        "messages": [
            AIMessage(content=message)
        ],
    }


def manager_review(state: ApprovalState) -> dict:
    """
    Node 3: Manager review with human-in-the-loop interrupt.

    TODO (3 points):
    - Call interrupt() with a dict containing request details for the manager:
      {"type": "manager_review", "request_id": ..., "title": ..., "amount": ...,
       "department": ..., "risk_level": ..., "risk_reasoning": ...}
    - The interrupt() return value is the manager's decision dict:
      {"approved": bool, "comments": str}
    - Store the decision in manager_approved and manager_comments
    - Record the decision in the decisions list
    - Add an AIMessage with the manager's decision
    - Return updated state fields

    Hints:
    - decision = interrupt({...}) pauses the graph until user responds
    - The frontend sends the decision as a JSON string — use json.loads(decision) if it's a str
    - Append to decisions list: {"stage": "manager_review", "approved": ...,
      "decision": "approved" | "rejected", "reviewer": "Manager", "comments": ...,
      "reasoning": ...}  ("decision" and "reasoning" are the keys the Part 5 audit-trail
      evaluator checks)
    - ApprovalState.decisions has no reducer, so return the whole list:
      {"decisions": state.get("decisions", []) + [entry]}; returning [entry] alone
      overwrites the earlier decisions (same for finance_review and final_signoff)
    - The route_after_manager router decides the next step based on approval + risk_level
    """
    # 1. Extract manager comments, if available
    manager_comments = state.get("manager_comments") or ""

    
    notify_reviewer(state, "Manager")

    # 2. Pause execution and request Finance approval
    response = interrupt({
        "type": "manager_review",
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "department": state.get("department"),
        "risk_level": state.get("risk_level"),
        "risk_reasoning": state.get("risk_reasoning"),
    })

    if isinstance(response, str):
      response = json.loads(response)

    

    # 3. Read the human decision
    approved = response.get("approved", False)
    comments = response.get("comments", "")

     

    # 4. Record the Finance decision
    decision = {
      "stage": "manager_review",
      "approved": approved,
      "decision": "approved" if approved else "rejected",
      "reviewer": "Manager",
      "comments": comments,
      "comment": comments,
      "reasoning": comments or "Manager decision recorded",
    }

    log_audit_event(
      request_id=state.get("request_id"),
      stage="manager_review",
      reviewer="Manager",
      decision="approved" if approved else "rejected",
      comments=comments,
    ) 

    # 5. Create decision message
    decision_text = "Approved" if approved else "Rejected"

    message = (
        f"Manager Review: {decision_text}\n"
        f"Comments: {comments or 'No comments provided.'}"
    )

    # 6. Return updated state
    return {
        "manager_approved": approved,
        "manager_comments": comments,
        "decisions": (state.get("decisions") or []) + [decision],
        "current_stage": "manager_review",
        "messages": [
            AIMessage(content=message)
        ],
    }


def validate_budget(state: ApprovalState) -> dict:
    """
    Node 4: Validate request against department budget.

    TODO (1 point):
    - Look up the department's budget from DEPARTMENT_BUDGETS
    - Check if the requested amount is within budget
    - Set department_budget, budget_remaining, and within_budget
    - Add an AIMessage with budget validation results
    - Return updated state fields

    Hints:
    - budget = DEPARTMENT_BUDGETS.get(state["department"], 0)
    - within_budget = state["amount"] <= budget
    - budget_remaining = budget - state["amount"]
    - The route_after_budget router decides the next step:
      - Within budget → process_request (auto-approve)
      - Over budget + low risk → escalate to manager_review
      - Over budget + medium risk → escalate to finance_review
    """
    department = state["department"]
    amount = state["amount"]
    budget = DEPARTMENT_BUDGETS.get(department, 0)
    within_budget = amount <= budget
    budget_remaining = budget - amount

    if within_budget:
        message = (
            f"Budget validation passed for {department}.\n"
            f"Department budget: ${budget:,.2f}\n"
            f"Requested amount: ${amount:,.2f}\n"
            f"Remaining budget: ${budget_remaining:,.2f}\n"
            f"Request is within budget."
        )
    else:
        message = (
            f"Budget validation failed for {department}.\n"
            f"Department budget: ${budget:,.2f}\n"
            f"Requested amount: ${amount:,.2f}\n"
            f"Budget exceeded by: ${abs(budget_remaining):,.2f}\n"
            f"Request requires additional approval."
        )

    return {
        "department_budget": budget,
        "budget_remaining": budget_remaining,
        "within_budget": within_budget,
        "messages": [
            AIMessage(content=message)
        ],
    }    


def finance_review(state: ApprovalState) -> dict:
    """
    Node 5: Finance team review with HITL interrupt.
    """

    # 1. Extract manager comments, if available
    manager_comments = state.get("manager_comments") or ""

    notify_reviewer(state, "Finance")

    # 2. Pause execution and request Finance approval
    response = interrupt({
        "type": "finance_review",
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "department": state.get("department"),
        "risk_level": state.get("risk_level"),
        "within_budget": state.get("within_budget"),
        "department_budget": state.get("department_budget"),
        "budget_remaining": state.get("budget_remaining"),
        "manager_comments": manager_comments,
    })

    if isinstance(response, str):
          response = json.loads(response)

    # 3. Read the human decision
    approved = response.get("approved", False)
    comments = response.get("comments", "")

    # 4. Record the Finance decision
    decision = {
        "stage": "finance_review",
        "approved": approved,
        "decision": "approved" if approved else "rejected",
        "reviewer": "Finance",
        "comments": comments,
        "comment": comments,
        "reasoning": comments or "Finance decision recorded",
    }

    log_audit_event(
      request_id=state.get("request_id"),
      stage="finance_review",
      reviewer="Finance",
      decision="approved" if approved else "rejected",
      comments=comments,
    )

    # 5. Create decision message
    decision_text = "Approved" if approved else "Rejected"

    message = (
        f"Finance Review: {decision_text}\n"
        f"Comments: {comments or 'No comments provided.'}"
    )

    # 6. Return updated state
    return {
        "finance_approved": approved,
        "finance_comments": comments,
        "decisions": (state.get("decisions") or []) + [decision],
        "current_stage": "finance_review",
        "messages": [
            AIMessage(content=message)
        ],
    }


def final_signoff(state: ApprovalState) -> dict:
    """
    Node 6: Executive final sign-off with human-in-the-loop interrupt.

    TODO (3 points):
    - Call interrupt() with a dict summarizing the full approval chain:
      {"type": "final_signoff", "request_id": ..., "title": ..., "amount": ...,
       "department": ..., "risk_level": ..., "manager_approved": ...,
       "finance_approved": ..., "within_budget": ..., "decisions": ...}
    - Store the decision in final_approved and final_comments
    - Set current_stage to "processing" if approved, else "rejection"
    - Record the decision in the decisions list
    - Add an AIMessage with final decision
    - Return updated state fields
    """

    notify_reviewer(state, "Executive")

    response = interrupt({
        "type": "final_signoff",
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "department": state.get("department"),
        "risk_level": state.get("risk_level"),
        "manager_approved": state.get("manager_approved"),
        "finance_approved": state.get("finance_approved"),
        "within_budget": state.get("within_budget"),
        "decisions": state.get("decisions") or [],
    })

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            response = {}
    if not isinstance(response, dict):
        response = {}

    approved = bool(response.get("approved", False))
    comments = response.get("comments", "")
    decision = {
        "stage": "final_signoff",
        "approved": approved,
        "decision": "approved" if approved else "rejected",
        "reviewer": "Executive",
        "comments": comments,
        "comment": comments,
        "reasoning": comments,
    }

    log_audit_event(
      request_id=state.get("request_id"),
      stage="final_signoff",
      reviewer="Executive",
      decision="approved" if approved else "rejected",
      comments=comments,
    )
    message = (
        f"Executive Final Sign-off: {'Approved' if approved else 'Rejected'}\n"
        f"Comments: {comments or 'No comments provided.'}"
    )

    return {
        "final_approved": approved,
        "final_comments": comments,
        "decisions": (state.get("decisions") or []) + [decision],
        "current_stage": "processing" if approved else "rejection",
        "messages": [AIMessage(content=message)],
    }


def process_request(state: ApprovalState) -> dict:
    """
    Node 7: Process the fully-approved request.
    """

    # 1. Extract request details
    title = state.get("title", "Untitled Request")
    amount = state.get("amount", 0)
    department = state.get("department", "Unknown")

    # 2. Retrieve approval history
    decisions = state.get("decisions") or []

    # 3. Build approval summary
    summary = [
        "Financial Request Approved",
        f"Title: {title}",
        f"Amount: ${amount:,.2f}",
        f"Department: {department}",
        "",
        "Approval History:"
    ]

    if decisions:
        for index, decision in enumerate(decisions, start=1):
            if isinstance(decision, dict):
                stage = decision.get("stage", "Unknown")
                approved = decision.get("approved")
                comment = decision.get("comment", "No comment")

                result = (
                    "Approved" if approved is True
                    else "Rejected" if approved is False
                    else "Pending"
                )

                summary.append(
                    f"{index}. {stage}: {result} — {comment}"
                )
            else:
                summary.append(f"{index}. {decision}")
    else:
        summary.append(
            "Automatically approved without manual review."
        )

    # 4. Sanitize output to remove sensitive information
    summary_text = "\n".join(summary)
    safe_summary = sanitize_output(summary_text)

    # 5. Return updated state
    return {
        "status": "approved",
        "current_stage": "complete",
        "messages": [
            AIMessage(content=safe_summary)
        ],
    }


def handle_rejection(state: ApprovalState) -> dict:
    """
        Node 8: Handle a rejected request.
    """

    # 1. Extract request details
    title = state.get("title", "Untitled Request")
    amount = state.get("amount", 0)
    department = state.get("department", "Unknown")

    # 2. Retrieve approval decisions
    decisions = state.get("decisions") or []

    rejected_stage = "Unknown"
    rejection_reason = "No rejection reason provided."

    # 3. Find the rejecting stage and reason
    for decision in reversed(decisions):
        if not isinstance(decision, dict):
            continue

        if (
            decision.get("approved") is False
            or decision.get("decision") == "rejected"
        ):
            rejected_stage = decision.get(
                "stage", "Unknown"
            )

            rejection_reason = (
                decision.get("comment")
                or decision.get("reason")
                or "No rejection reason provided."
            )
            break

      # 4. Build rejection summary
    summary = (
        "Financial Request Rejected\n"
        f"Title: {title}\n"
        f"Amount: ${amount:,.2f}\n"
        f"Department: {department}\n"
        f"Rejected at: {rejected_stage}\n"
        f"Reason: {rejection_reason}"
    )

    # 5. Sanitize output to remove PII
    safe_summary = sanitize_output(summary)

    # 6. Return updated state
    return {
        "status": "rejected",
        "current_stage": "complete",
        "messages": [
            AIMessage(content=safe_summary)
        ],
    }


# ============================================================
# ROUTING FUNCTIONS (6 total — implement the escalation logic)
# ============================================================

def route_after_submission(state: ApprovalState) -> str:
    """
    Router: After submit_request, go to risk assessment or rejection.
    """

    if state.get("is_valid") is True:
        return "assess_risk"

    return "handle_rejection"


def route_after_risk(state: ApprovalState) -> str:
    """
    Router: After assess_risk, route based on risk level.

    This is the core escalation entry point. Low-risk requests skip
    human review and go directly to budget validation (auto-approve path).
    All other risk levels require manager review first.

    TODO (1 point):
    - Return "validate_budget" if risk_level is "low"
    - Return "manager_review" for "medium", "high", or "critical"

    Hints:
    - state["risk_level"] contains the assessed risk level
    - Low risk = auto-approve path (no human review needed if within budget)
    """
    if state.get("risk_level") == "low":
        return "validate_budget"
    return "manager_review"


def route_after_manager(state: ApprovalState) -> str:
    """
    Router: After manager_review, route based on approval + risk level.

    The next step depends on BOTH the manager's decision AND the risk level:
    - Rejected → handle_rejection (any risk level)
    - Approved + low risk (over-budget escalation) → process_request
    - Approved + medium risk → validate_budget
    - Approved + high/critical risk → finance_review

    TODO (1 point):
    - Return "handle_rejection" if state["manager_approved"] is not True
    - Return "process_request" if risk_level is "low" (was escalated from over-budget)
    - Return "validate_budget" if risk_level is "medium"
    - Return "finance_review" if risk_level is "high" or "critical"

    Hints:
    - Low-risk requests only reach manager_review when they are over budget.
      The manager already approved, so we can proceed to processing.
    - Medium-risk requests need budget validation after manager approval.
    - High/critical-risk requests skip budget check and go to finance.
    """
    if state.get("manager_approved") is not True:
        return "handle_rejection"
    risk_level = state.get("risk_level")
    if risk_level == "low":
        return "process_request"
    if risk_level == "medium":
        return "validate_budget"
    return "finance_review"


def route_after_budget(state: ApprovalState) -> str:
    """
    Router: After validate_budget, route based on budget status + risk level.

    Within-budget requests proceed to processing (auto-approved).
    Over-budget requests escalate to the next human reviewer.

    TODO (1 point):
    - Return "process_request" if state["within_budget"] is True
    - If over budget:
      - Return "manager_review" if risk_level is "low" (needs human review)
      - Return "finance_review" if risk_level is "medium" (manager already approved)

    Hints:
    - Low-risk over-budget: first time seeing a human → escalate to manager
    - Medium-risk over-budget: manager already approved → escalate to finance
    - High/critical-risk requests don't go through this router (they skip budget check)
    """
    if state.get("within_budget") is True:
        return "process_request"
    if state.get("risk_level") == "low":
        return "manager_review"
    return "finance_review"


def route_after_finance(state: ApprovalState) -> str:
    """
    Router: After finance_review, route based on approval + risk level.

    The next step depends on BOTH finance's decision AND the risk level:
    - Rejected → handle_rejection
    - Approved + critical risk → final_signoff (one more review)
    - Approved + any other risk → process_request (done)

    TODO (1 point):
    - Return "handle_rejection" if state["finance_approved"] is not True
    - Return "final_signoff" if risk_level is "critical"
    - Return "process_request" otherwise
    """
    if state.get("finance_approved") is not True:
        return "handle_rejection"
    if state.get("risk_level") == "critical":
        return "final_signoff"
    return "process_request"


def route_after_final(state: ApprovalState) -> str:
    """
    Router: After final_signoff, go to processing or rejection.

    TODO (1 point):
    - Return "process_request" if state["final_approved"] is True
    - Return "handle_rejection" otherwise
    """
    if state.get("final_approved") is True:
        return "process_request"
    return "handle_rejection"


def parallel_manager_review(
    state: ApprovalState,
    config: RunnableConfig,
):
    """Manager review for the parallel high-risk path."""

    thread_id = config.get("configurable", {}).get("thread_id")


    print(
      f"[PARALLEL] Manager review started "
      f"for {state.get('request_id')}",
      flush=True,
    )
    print("[PARALLEL MANAGER] Actual thread:", thread_id)


    response = interrupt({
        "type": "manager_review",
        "thread_id": thread_id,
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "department": state.get("department"),
        "risk_level": state.get("risk_level"),
        "risk_reasoning": state.get("risk_reasoning"),
    })

    if isinstance(response, str):
        response = json.loads(response)

    if not isinstance(response, dict):
        response = {}

    approved = response.get("approved") is True
    comments = str(response.get("comments") or "")

    decision = {
        "stage": "manager_review",
        "approved": approved,
        "decision": "approved" if approved else "rejected",
        "reviewer": "Manager",
        "comments": comments,
        "comment": comments,
        "reasoning": comments or "Manager decision recorded",
    }

    log_audit_event(
      request_id=state.get("request_id"),
      stage="manager_review",
      reviewer="Manager",
      decision="approved" if approved else "rejected",
      comments=comments,
    )

    return {
        "parallel_manager_decision": decision,
        "messages": [
            AIMessage(
                content=f"Parallel Manager Review: "
                        f"{'Approved' if approved else 'Rejected'}"
            )
        ],
    }


def parallel_finance_review(state: ApprovalState, config: RunnableConfig) -> dict:
    """Finance review for the parallel high-risk path."""

    thread_id = config.get("configurable", {}).get("thread_id")

    print(
      f"[PARALLEL] Finance review started "
      f"for {state.get('request_id')}",
      flush=True,
    )

    print("[PARALLEL FINANCE] Actual thread:", thread_id)

    response = interrupt({
        "type": "finance_review",
        "thread_id": thread_id,
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "department": state.get("department"),
        "risk_level": state.get("risk_level"),
        "risk_reasoning": state.get("risk_reasoning"),
        "within_budget": state.get("within_budget"),
        "department_budget": state.get("department_budget"),
        "budget_remaining": state.get("budget_remaining"),
        "manager_comments": state.get("manager_comments") or "",
    })

    if isinstance(response, str):
        response = json.loads(response)

    if not isinstance(response, dict):
        response = {}

    approved = response.get("approved") is True
    comments = str(response.get("comments") or "")

    decision = {
        "stage": "finance_review",
        "approved": approved,
        "decision": "approved" if approved else "rejected",
        "reviewer": "Finance",
        "comments": comments,
        "comment": comments,
        "reasoning": comments or "Finance decision recorded",
    }

    log_audit_event(
      request_id=state.get("request_id"),
      stage="finance_review",
      reviewer="Finance",
      decision="approved" if approved else "rejected",
      comments=comments,
    ) 

    return {
        "parallel_finance_decision": decision,
        "messages": [
            AIMessage(
                content=f"Parallel Finance Review: "
                        f"{'Approved' if approved else 'Rejected'}"
            )
        ],
    }


def join_parallel_reviews(state: ApprovalState) -> dict:
    """Merge manager and finance decisions."""

    manager = state.get("parallel_manager_decision")
    finance = state.get("parallel_finance_decision")

    print(
          f"[PARALLEL] Joining parallel reviews "
          f"for {state.get('request_id')}",
          flush=True,
    )

    if not isinstance(manager, dict) or not isinstance(finance, dict):
        raise ValueError(
            "Both parallel reviews must finish before joining"
        )

    return {
        "manager_approved": manager["approved"],
        "manager_comments": manager["comments"],
        "finance_approved": finance["approved"],
        "finance_comments": finance["comments"],
        "decisions": (
            (state.get("decisions") or []) + [manager, finance]
        ),
        "current_stage": "parallel_review_complete",
    }


def route_after_parallel_reviews(state: ApprovalState) -> str:
    """Require approval from both reviewers."""

    if (
        state.get("manager_approved") is True
        and state.get("finance_approved") is True
    ):
        return "process_request"

    return "handle_rejection"
