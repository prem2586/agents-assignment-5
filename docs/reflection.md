# Assignment 5: Reflection

## Part 1: LangGraph Workflow

### 1. How did you design the 8-node workflow? What considerations influenced the order of approval stages?

I designed the LangGraph workflow to represent a real-world financial approval process, where requests pass through validation, risk assessment, budget verification, human approval, and final processing.

The workflow follows a logical sequence in which each stage determines whether the request should proceed to the next stage. Input validation happens early to prevent invalid requests from entering the approval process. Risk assessment helps determine the appropriate approval path, while budget validation and human reviews provide additional controls before financial requests are finalized.

The main design considerations were separation of responsibilities, conditional routing, auditability, and human oversight. I also wanted to avoid unnecessary approval steps for lower-risk requests while ensuring that higher-risk requests receive additional scrutiny.

### 2. Describe your experience implementing the `interrupt()` pattern. What was intuitive and what was challenging?

The `interrupt()` pattern was one of the most interesting parts of the assignment because it allowed me to introduce human decision-making into an automated LangGraph workflow.

The intuitive part was that `interrupt()` pauses execution and provides the information required for a reviewer to make a decision. Once the reviewer responds, `Command(resume=...)` allows the workflow to continue.

The challenging part was managing execution state and thread IDs, particularly when implementing parallel approvals.

Initially, the frontend was passing the financial request ID instead of the actual LangGraph execution thread ID. This caused the backend to fail to locate the pending interrupts. I resolved the issue by retrieving the actual thread ID from `RunnableConfig` and including it in the interrupt payload.

Another challenge was handling multiple simultaneous interrupts. I learned that resuming parallel approvals requires mapping each interrupt ID to its corresponding decision.

This experience improved my understanding of LangGraph checkpointing, execution state, and human-in-the-loop orchestration.

### 3. How do the routing functions control the workflow? Why are conditional edges important for approval systems?

Routing functions evaluate the current workflow state and determine which node should execute next.

For example, a request's risk classification can determine whether it requires additional validation, manager approval, finance approval, or a higher level of review.

Conditional edges are important because financial approval processes are not always linear. Different requests require different controls depending on their risk level, amount, and previous decisions.

They also allow the workflow to terminate or redirect processing when a request is rejected.

Using conditional routing makes the system more efficient, maintainable, and aligned with real business approval policies.

## Part 2: Safety Guardrails

### 1. What types of attacks does your input validation protect against? Are there any gaps?

Input validation is intended to protect the workflow against malformed requests, unexpected data types, missing required fields, and potentially malicious content.

For an AI-powered financial approval system, important risks include prompt injection, manipulation of request descriptions, and attempts to bypass approval rules.

Validation helps ensure that financial amounts, departments, and other structured fields meet expected requirements before the workflow processes them.

However, input validation alone cannot eliminate every risk. Sophisticated prompt injections may appear within otherwise valid business descriptions.

In production, I would strengthen these protections with authorization checks, strict request schemas, deterministic business-rule enforcement, rate limiting, and security monitoring.

### 2. How did you implement PII masking? What trade-offs exist between security and usability?

The purpose of PII masking is to prevent sensitive personal information from being unnecessarily exposed in logs, AI model interactions, and user-facing responses.

The intended approach is to identify sensitive values and replace them with masked representations before exposing the data to components that do not require the original information.

For example, an email address could be partially masked while retaining enough information for a reviewer to recognize the record.

The main trade-off is between privacy and usability. Excessive masking can make it difficult for reviewers to verify information, while insufficient masking increases the risk of exposing sensitive data.

A production implementation should use role-based access controls so that authorized users can access necessary information without exposing it to everyone.

### 3. Why is output filtering important in financial systems, even when inputs are validated?

Output filtering provides an additional security layer because validated inputs do not guarantee that an AI model will produce safe or accurate outputs.

An AI model might accidentally reveal sensitive information, generate unsupported financial conclusions, or produce content inconsistent with business policies.

Output filtering helps ensure that responses follow expected formats, avoid exposing sensitive information, and do not make unauthorized financial decisions.

I consider output filtering an important part of defense-in-depth security, especially when AI-generated recommendations influence financial approval decisions.

## Part 3: CopilotKit Frontend

### 1. How does the `useLangGraphInterrupt` hook connect the frontend to the backend workflow?

The `useLangGraphInterrupt` hook allows the React frontend to respond when a LangGraph workflow pauses for human input.

When an interrupt occurs, the hook receives the interrupt event containing relevant request details, such as the request ID, amount, department, risk level, and approval stage.

The frontend renders an approval interface using this information.

For sequential approvals, the reviewer submits a decision through the `resolve()` callback, allowing CopilotKit to resume the backend workflow.

For the parallel approval bonus, I implemented a separate FastAPI endpoint to resume multiple pending interrupts together.

### 2. Describe the user experience when an approval interrupt fires. How did you design the UI to be clear and actionable?

When an approval interrupt occurs, the frontend displays a modal containing the financial request details.

The interface presents the request title, amount, department, risk classification, and relevant assessment information.

Reviewers can choose Approve or Reject and provide optional comments.

For high-risk requests, I implemented a parallel approval modal that collects separate decisions from Manager and Finance reviewers.

The submission button remains disabled until both decisions are selected, helping prevent incomplete submissions.

I focused on making the UI straightforward, with clearly labeled actions, visible risk information, and feedback during submission.

### 3. What challenges did you face integrating CopilotKit with the LangGraph backend?

The biggest challenge was synchronizing the frontend with LangGraph's checkpointed execution state.

Initially, the frontend and backend used different thread identifiers, preventing the backend from finding the correct pending interrupts.

Another challenge occurred when both Manager and Finance approval nodes were interrupted simultaneously. The standard single-interrupt resolution approach was insufficient because LangGraph required explicit interrupt IDs.

I addressed this by creating a FastAPI endpoint that retrieves pending interrupts from the graph state and resumes them using an interrupt-ID-to-decision mapping.

I also encountered a frontend synchronization issue: the backend successfully processed approval decisions, but the modal and workflow display did not automatically reflect the updated state.

This highlighted an important distinction between successfully updating backend workflow state and keeping the frontend synchronized with those changes.

## Part 4: LangSmith Evaluation

### 1. How does your risk accuracy evaluator handle partial matches (adjacent risk levels)?

A risk accuracy evaluator can compare the predicted risk classification with the expected classification.

An exact match receives full credit, while an adjacent risk level can receive partial credit because the prediction is close to the expected result.

For example, predicting High when the expected classification is Critical is less severe than predicting Low for a Critical request.

A possible scoring approach is:

- Exact match: 1.0
- Adjacent risk level: 0.5
- More distant classification: 0.0

This approach provides more meaningful evaluation than a simple correct-or-incorrect comparison.

However, the evaluator should also distinguish underestimating risk from overestimating risk, because underestimation may have greater consequences in a financial approval system.

### 2. Why is evaluation important for production AI systems? How would you extend the evaluation suite?

Evaluation is essential because AI-generated decisions are probabilistic and may behave differently across inputs, model versions, and prompt changes.

In a financial approval workflow, incorrect risk classification can lead to unnecessary approvals, missed reviews, or inappropriate rejection of legitimate requests.

LangSmith evaluation can help measure risk classification accuracy, identify edge cases, and detect regressions when the workflow changes.

I would extend the evaluation suite to include prompt-injection tests, invalid financial amounts, boundary-value testing, PII leakage detection, routing accuracy, approval-policy compliance, and human-in-the-loop recovery scenarios.

I would also test parallel approvals with different combinations of Manager and Finance decisions.

### 3. What insights did the evaluation results reveal about your workflow's behavior?

The evaluation process helped me understand the importance of testing not only AI-generated classifications but also the downstream business decisions that depend on them.

Risk classification is especially important because it determines which approval stages a request enters.

I also recognized that boundary cases between adjacent risk levels require careful evaluation, since small classification differences can change the workflow path.

An important learning was that evaluation should consider both model accuracy and the correctness of the overall workflow.

*The specific LangSmith scores and observed failure cases should be added from the actual evaluation run before submission.*

## Bonus Features

### Parallel Manager and Finance Approval

I implemented parallel approval processing for high-risk financial requests using LangGraph.

Instead of executing Manager and Finance reviews sequentially, the workflow starts both approval nodes concurrently.

Each node uses `interrupt()` to pause execution until a human decision is available.

A custom FastAPI endpoint retrieves the pending interrupt IDs and resumes both reviews using their corresponding decisions.

After both decisions are received, a join node combines the results and determines the next workflow action.

The join logic requires both Manager and Finance approval for the request to proceed. If either reviewer rejects the request, the combined outcome is rejected.

I verified this behavior through backend execution logs, including scenarios where both reviewers rejected the request.

This feature demonstrated how LangGraph can support more complex approval patterns involving parallel human decisions and state synchronization.

### Streamlit Audit Dashboard

I also worked with a Streamlit dashboard that reads approval events from SQLite.

The dashboard includes total decisions, approved and rejected counts, approval rate, reviewer distribution, and recent approval activity.

During testing, I discovered that decisions successfully recorded in LangGraph state were not necessarily appearing in the Streamlit audit dashboard.

This helped me understand the difference between workflow checkpoint persistence and application-level audit logging.

The remaining integration work is to ensure that all parallel approval decisions are written to the audit database and consistently displayed in Streamlit.

## Key Learnings

- **Human-in-the-loop orchestration:** I learned how LangGraph interrupts, checkpointing, and `Command(resume=...)` enable human approvals within an AI workflow.
- **State management across systems:** I gained practical experience integrating LangGraph, FastAPI, CopilotKit, React, and SQLite, particularly handling thread IDs, parallel interrupts, and frontend synchronization.
- **Production AI engineering:** I learned that successful AI workflows require more than model execution. They also need validation, safety controls, evaluation, auditability, and reliable handling of business decisions.

## Ideas for Improvement

- **Improve end-to-end synchronization and auditability:** Ensure that all approved and rejected decisions are consistently persisted, reflected in the Streamlit dashboard, and synchronized with the CopilotKit frontend.
- **Strengthen production readiness:** Add role-based reviewer authorization, idempotent approval submission, comprehensive automated testing, improved error handling, and additional LangSmith evaluation scenarios.