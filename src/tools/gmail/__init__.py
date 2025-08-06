"""Gmail tools for email assistant."""

from src.tools.gmail.gmail_tools import (
    fetch_emails_tool,
    send_email_tool,
    check_calendar_tool,
    schedule_meeting_tool
)

from src.tools.gmail.prompt_templates import GMAIL_TOOLS_PROMPT

__all__ = [
    "fetch_emails_tool",
    "send_email_tool",
    "check_calendar_tool",
    "schedule_meeting_tool",
    "GMAIL_TOOLS_PROMPT"
]