
import json
from datetime import datetime, timezone
from pathlib import Path

NOTIFICATION_LOG = (
    Path(__file__).resolve().parent / "notifications.jsonl"
)


def notify_reviewer(state, review_type):
    """Simulate an email and Slack review notification."""

    notification = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": state.get("request_id"),
        "title": state.get("title"),
        "amount": state.get("amount"),
        "review_type": review_type,
        "status": "notification_simulated",
        "channels": ["email", "slack"],
    }

    with open(NOTIFICATION_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(notification) + "\n")

    print(
        f"\n[EMAIL / SLACK NOTIFICATION]\n"
        f"Request: {notification['request_id']}\n"
        f"Title: {notification['title']}\n"
        f"Amount: ${notification['amount']:,.2f}\n"
        f"Review required: {review_type}\n"
    )

    return notification
