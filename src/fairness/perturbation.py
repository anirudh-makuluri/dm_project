from __future__ import annotations

import re
from dataclasses import dataclass


NAME_SWAPS = (
    ("John Doe", "Aisha Khan"),
    ("Jane Smith", "Miguel Santos"),
    ("Alex Johnson", "Priya Patel"),
    ("Maria Garcia", "Darnell Brooks"),
    ("Kevin Brown", "Wei Chen"),
    ("Ravi Patel", "Elena Petrova"),
    ("Emily Davis", "Omar Ali"),
    ("Samuel Lee", "Fatima Rahman"),
    ("Linda Nguyen", "Noah Williams"),
    ("Anika Sharma", "Grace Miller"),
)

PRONOUN_MAP = {
    r"\bhe\b": "she",
    r"\bshe\b": "he",
    r"\bhim\b": "her",
    r"\bher\b": "him",
    r"\bhis\b": "her",
    r"\bhers\b": "his",
    r"\bhimself\b": "herself",
    r"\bherself\b": "himself",
    r"\bHe\b": "She",
    r"\bShe\b": "He",
    r"\bHim\b": "Her",
    r"\bHer\b": "Him",
    r"\bHis\b": "Her",
    r"\bHers\b": "His",
    r"\bHimself\b": "Herself",
    r"\bHerself\b": "Himself",
}


@dataclass(frozen=True)
class PerturbationVariant:
    variant_type: str
    text: str
    metadata: dict[str, str]


def swap_pronouns(text: str) -> str:
    swapped = text
    for pattern, replacement in PRONOUN_MAP.items():
        swapped = re.sub(pattern, replacement, swapped)
    return swapped


def swap_name(text: str, seed: int = 0) -> str:
    for original, replacement in NAME_SWAPS[seed % len(NAME_SWAPS) :]:
        if original in text:
            return text.replace(original, replacement, 1)
    for original, replacement in NAME_SWAPS[: seed % len(NAME_SWAPS) + 1]:
        if original in text:
            return text.replace(original, replacement, 1)
    first_line, *rest = text.splitlines()
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", first_line.strip()):
        return text.replace(first_line, NAME_SWAPS[seed % len(NAME_SWAPS)][1], 1)
    return text


def inject_identity_cue(text: str, cue: str = "from an underrepresented background") -> str:
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    if first and cue not in text:
        lines.insert(1, f"Identity cue: {cue}.")
    return "\n".join(lines)


def build_perturbations(text: str, seed: int = 0) -> list[PerturbationVariant]:
    variants = [
        PerturbationVariant("original", text, {"source": "original"}),
        PerturbationVariant("name_swap", swap_name(text, seed=seed), {"source": "name_swap"}),
        PerturbationVariant("pronoun_swap", swap_pronouns(text), {"source": "pronoun_swap"}),
        PerturbationVariant(
            "combined",
            inject_identity_cue(swap_pronouns(swap_name(text, seed=seed))),
            {"source": "combined"},
        ),
    ]
    unique: list[PerturbationVariant] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.text.strip()
        if key and key not in seen:
            unique.append(variant)
            seen.add(key)
    return unique
