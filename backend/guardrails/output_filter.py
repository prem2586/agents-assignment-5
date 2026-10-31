"""
Output filtering guardrails for the Financial Approval System.

Students implement output sanitization that masks PII and
sensitive financial details before responses reach users.

Part 2: Safety Guardrails (10 points from this file)
"""

import re

# --- GIVEN: PII detection patterns ---
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
}


def sanitize_output(text: str) -> str:
    """
    Remove PII and sensitive information from output text.

    TODO (5 points):
    - For each pattern in PII_PATTERNS:
      - "ssn": replace matches with "***-**-XXXX" (preserve nothing)
      - "credit_card": replace matches with "****-****-****-XXXX"
      - "email": replace matches with "[EMAIL REDACTED]"
      - "phone": replace matches with "***-***-XXXX"
    - Return the sanitized text

    Args:
        text: Output text that may contain PII

    Returns:
        Sanitized text with PII masked
    """
    def mask_ssn(match: re.Match) -> str:
        return "***-**-XXXX"

    def mask_credit_card(match: re.Match) -> str:
        return "****-****-****-XXXX"

    def mask_email(match: re.Match) -> str:
        return "[EMAIL REDACTED]"

    def mask_phone(match: re.Match) -> str:
        return "***-***-XXXX"

    text = PII_PATTERNS["ssn"].sub(mask_ssn, text)
    text = PII_PATTERNS["credit_card"].sub(mask_credit_card, text)
    text = PII_PATTERNS["email"].sub(mask_email, text)
    text = PII_PATTERNS["phone"].sub(mask_phone, text)

    return text


def mask_financial_details(text: str) -> str:
    r"""
    Mask specific financial details in output while preserving context.

    TODO (5 points):
    - Find dollar amounts matching pattern $X,XXX.XX or $XXXXX.XX
    - Replace with masked version showing only last 3 digits: $***XX.XX
    - Find account numbers (8+ consecutive digits) and mask all but last 4
    - Return the masked text

    Hints:
    - Use re.sub with a replacement function for dollar amounts
    - Pattern for dollars: r'\$[\d,]+\.?\d*'
    - Pattern for account numbers: r'\b\d{8,}\b'

    Args:
        text: Text containing financial details

    Returns:
        Text with financial details masked
    """
    def mask_amount(match: re.Match) -> str:
        amount = match.group()[1:]
        integer, separator, decimals = amount.partition(".")
        digits = integer.replace(",", "")
        visible = digits[-2:]
        masked_integer = "*" * max(3, len(digits) - len(visible)) + visible
        return f"${masked_integer}{separator}{decimals}"

    text = re.sub(r"\$[\d,]+\.?\d*", mask_amount, text)
    return re.sub(
        r"\b\d{8,}\b",
        lambda match: "*" * (len(match.group()) - 4) + match.group()[-4:],
        text,
    )
