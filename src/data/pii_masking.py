from __future__ import annotations

import re

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
LINKEDIN_RE = re.compile(r"\b(?:https?://)?(?:www\.)?linkedin\.com/\S+\b", re.IGNORECASE)
GITHUB_RE = re.compile(r"\b(?:https?://)?(?:www\.)?github\.com/\S+\b", re.IGNORECASE)


def mask_pii(text: str) -> str:
    masked = text
    masked = EMAIL_RE.sub("[EMAIL]", masked)
    masked = PHONE_RE.sub("[PHONE]", masked)
    masked = LINKEDIN_RE.sub("[LINKEDIN]", masked)
    masked = GITHUB_RE.sub("[GITHUB]", masked)
    masked = URL_RE.sub("[URL]", masked)
    return masked
