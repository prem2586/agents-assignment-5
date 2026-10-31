"use client";

import { useState } from "react";
import {
  useLangGraphInterrupt,
  useThreads,
} from "@copilotkit/react-core";
import { InterruptData, ApprovalDecision } from "@/lib/types";

const STAGE_LABELS: Record<string, string> = {
  manager_review: "Manager Review",
  finance_review: "Finance Review",
  final_signoff: "Final Sign-off",
};

const RISK_COLORS: Record<string, string> = {
  low: "text-green-600 bg-green-100",
  medium: "text-yellow-600 bg-yellow-100",
  high: "text-orange-600 bg-orange-100",
  critical: "text-red-600 bg-red-100",
};

function ApprovalModal({
  data,
  resolve,
}: {
  data: InterruptData;
  resolve: (resolution: string) => void;
}) {
  const [comments, setComments] = useState("");
  const stageLabel = STAGE_LABELS[data.type] || data.type;
  const riskColor = RISK_COLORS[data.risk_level] || "text-gray-600 bg-gray-100";

  const handleDecision = (approved: boolean) => {
    resolve(JSON.stringify({ approved, comments }));
    setComments("");
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-blue-600 text-white px-6 py-4 rounded-t-xl">
          <h2 className="text-xl font-bold">{stageLabel}</h2>
          <p className="text-blue-100 text-sm mt-1">
            Review the request details and provide your decision
          </p>
        </div>

        <div className="p-6 space-y-4">
          {/* Request Details */}
          <div className="space-y-3">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">Request</p>
                <p className="font-semibold text-gray-900">{data.title}</p>
              </div>
              <span className="text-xs font-mono text-gray-400">{data.request_id}</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">Amount</p>
                <p className="text-lg font-bold text-gray-900">
                  ${data.amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">Department</p>
                <p className="font-medium text-gray-900 capitalize">{data.department}</p>
              </div>
            </div>

            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide">Risk Level</p>
              <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold mt-1 ${riskColor}`}>
                {data.risk_level?.toUpperCase()}
              </span>
            </div>

            {data.risk_reasoning && (
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wide">Risk Assessment</p>
                <p className="text-sm text-gray-700 mt-1">{data.risk_reasoning}</p>
              </div>
            )}
          </div>

          {/* Budget Info (finance_review) */}
          {data.type === "finance_review" && (
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <p className="text-sm font-semibold text-gray-700">Budget Information</p>
              {/* High and critical requests skip validate_budget, so these fields can be missing */}
              {data.department_budget == null ? (
                <p className="text-sm text-gray-500">
                  Not checked: high and critical requests skip Budget Validation.
                </p>
              ) : (
              <>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-500">Department Budget:</span>
                  <span className="ml-1 font-medium">
                    ${data.department_budget?.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Remaining:</span>
                  <span className={`ml-1 font-medium ${(data.budget_remaining ?? 0) < 0 ? "text-red-600" : "text-green-600"}`}>
                    ${data.budget_remaining?.toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className={`w-3 h-3 rounded-full ${data.within_budget ? "bg-green-500" : "bg-red-500"}`} />
                <span className="text-sm font-medium">
                  {data.within_budget ? "Within budget" : "Exceeds budget"}
                </span>
              </div>
              </>
              )}
              {data.manager_comments && (
                <div className="mt-2 pt-2 border-t">
                  <p className="text-xs text-gray-500">Manager Comments:</p>
                  <p className="text-sm text-gray-700 italic">{data.manager_comments}</p>
                </div>
              )}
            </div>
          )}

          {/* Previous Decisions (final_signoff) */}
          {data.type === "final_signoff" && data.decisions && data.decisions.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <p className="text-sm font-semibold text-gray-700">Previous Decisions</p>
              {data.decisions.map((decision: ApprovalDecision, idx: number) => (
                <div key={idx} className="flex items-center gap-2 text-sm">
                  <span className={`w-2.5 h-2.5 rounded-full ${decision.approved ? "bg-green-500" : "bg-red-500"}`} />
                  <span className="font-medium">{decision.stage}:</span>
                  <span className={decision.approved ? "text-green-700" : "text-red-700"}>
                    {decision.approved ? "Approved" : "Rejected"}
                  </span>
                  {decision.comments && (
                    <span className="text-gray-500 truncate">- {decision.comments}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Comments */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Review Comments
            </label>
            <textarea
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Add your review comments..."
              className="w-full p-3 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              rows={3}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={() => handleDecision(true)}
              className="flex-1 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
            >
              Approve
            </button>
            <button
              onClick={() => handleDecision(false)}
              className="flex-1 py-3 bg-red-600 text-white rounded-lg font-semibold hover:bg-red-700 transition-colors"
            >
              Reject
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/*export default function ApprovalPanel() {
  useLangGraphInterrupt({
    render: ({ event, resolve }) => {
      const data = event.value as InterruptData;
      return <ApprovalModal data={data} resolve={resolve} />;
    },
  });

  return null;
}
 */

const submitParallelApprovals = async (
  threadId: string,
  managerDecision: { approved: boolean; comments: string },
  financeDecision: { approved: boolean; comments: string }
) => {
  const response = await fetch(
    "http://localhost:8000/parallel-approval",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        thread_id: threadId,
        manager: managerDecision,
        finance: financeDecision,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Parallel approval failed");
  }

  return response.json();
};

function ParallelApprovalModal({
  data,
  threadId,
}: {
  data: InterruptData;
  threadId: string;
}) {
  const [managerApproved, setManagerApproved] = useState<boolean | null>(null);
  const [financeApproved, setFinanceApproved] = useState<boolean | null>(null);
  const [managerComments, setManagerComments] = useState("");
  const [financeComments, setFinanceComments] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const submit = async () => {
    if (
      managerApproved === null ||
      financeApproved === null ||
      submitting ||
      submitted
    ) return;

    setSubmitting(true);
    setError("");
    try {
      const threadId = (
      data as InterruptData & { thread_id?: string }
      ).thread_id;

        if (!threadId) {
          throw new Error(
            "LangGraph thread ID is missing. Cannot submit parallel approvals."
          );
        }

        const result = await submitParallelApprovals(
          threadId,
          {
            approved: managerApproved,
            comments: managerComments,
          },
          {
            approved: financeApproved,
            comments: financeComments,
          }
        );

        console.log("Parallel approval result:", result);

        window.dispatchEvent(
          new CustomEvent("approval-completed", {
            detail: result,
          })
        );

        setSubmitted(true);
      } catch (error) {
        console.error("Parallel approval error:", error);

        setError(
          error instanceof Error
            ? error.message
            : "Unable to submit parallel approvals."
        );
          } finally {
        setSubmitting(false);
          }
  };

  const decisionButtons = (
    label: string,
    value: boolean | null,
    onChange: (approved: boolean) => void
  ) => (
    <div>
      <p className="text-sm font-semibold text-gray-700 mb-2">{label}</p>
      <div className="flex gap-3">
        <button
          type="button"
          aria-pressed={value === true}
          onClick={() => onChange(true)}
          className={`flex-1 py-2 rounded-lg font-semibold ${value === true ? "bg-green-600 text-white" : "bg-green-100 text-green-700"}`}
        >
          Approve
        </button>
        <button
          type="button"
          aria-pressed={value === false}
          onClick={() => onChange(false)}
          className={`flex-1 py-2 rounded-lg font-semibold ${value === false ? "bg-red-600 text-white" : "bg-red-100 text-red-700"}`}
        >
          Reject
        </button>
      </div>
    </div>
  );

  if (submitted) {
   return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
        <div className="bg-orange-600 text-white px-6 py-4 rounded-t-xl">
          <h2 className="text-xl font-bold">Parallel Approval</h2>
          <p className="text-orange-100 text-sm mt-1">Manager and Finance decisions are required</p>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Request</p>
            <p className="font-semibold text-gray-900">{data.title}</p>
            <p className="text-sm text-gray-600">
              {data.request_id} · ${data.amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </p>
          </div>
          {decisionButtons("Manager Decision", managerApproved, setManagerApproved)}
          <textarea
            aria-label="Manager comments"
            value={managerComments}
            onChange={(event) => setManagerComments(event.target.value)}
            placeholder="Manager comments..."
            className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none"
            rows={2}
          />
          {decisionButtons("Finance Decision", financeApproved, setFinanceApproved)}
          <textarea
            aria-label="Finance comments"
            value={financeComments}
            onChange={(event) => setFinanceComments(event.target.value)}
            placeholder="Finance comments..."
            className="w-full p-3 border border-gray-300 rounded-lg text-sm resize-none"
            rows={2}
          />
          {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
          <button
            type="button"
            onClick={submit}
            disabled={managerApproved === null || financeApproved === null || submitting || submitted}
            className="w-full py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting
              ? "Submitting..."
              : submitted
                ? "Decisions Submitted"
                : "Submit Both Decisions"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ApprovalPanel() {
  const { threadId } = useThreads();

  console.log("[ApprovalPanel] CopilotKit threadId:", threadId);

  useLangGraphInterrupt({
    render: ({ event, resolve }) => {
      const data = event.value as InterruptData;

      if (
        data.risk_level === "high" &&
        ["manager_review", "finance_review"].includes(data.type)
      ) {
        return (
          <ParallelApprovalModal
            key={`${data.request_id}-parallel`}
            data={data}
            threadId={threadId}
          />
        );
      }

      return (
        <ApprovalModal
          key={`${data.request_id}-${data.type}`}
          data={data}
          resolve={resolve}
        />
      );
    },
  });

  return null;
}
