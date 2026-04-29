"""Prompt templates used by the LLM-backed agents.

Each prompt asks the model to return JSON matching the corresponding
schema in :mod:`src.llm.schemas`. The agents call ``provider.complete``
and then validate the response with Pydantic. Any parse / validation
failure routes the agent to its deterministic fallback so the rest of
the pipeline keeps running.
"""

from __future__ import annotations

EXTRACTOR_PROMPT = """You are a resume parser. Extract structured fields
from the resume text below. Return ONLY a JSON object with these keys
(arrays of short strings):

- education: schools, degrees, graduation years.
- experience: bullet points of professional / work experience (one item per role or bullet).
- skills_raw: phrases naming a tool, technology, or competency (do not invent items not in the resume).
- projects: notable projects with a 1-line summary each.
- certifications: certifications, licenses, or credentials.

Rules:
- Use the resume text verbatim where possible. Do not paraphrase skill names.
- Omit empty fields rather than inventing entries.
- Output JSON only. No prose, no markdown fences.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


SKILL_MINER_PROMPT = """You are a skill mining agent. Identify the
professional skills mentioned in the resume below. Return ONLY a JSON object:

{{"phrases": ["...", "..."]}}

INCLUDE phrases that name:
- Specific tools, software, libraries, languages, frameworks (Excel, Python, React, AWS, Salesforce, SQL, ...)
- Methodologies / processes (Agile, Scrum, DevOps, ETL, project management, ...)
- Domain skills (financial modeling, recruiting, supply chain, graphic design, ...)
- Professional competencies the resume explicitly lists as skills (leadership, communication, ...)

EXCLUDE:
- Spoken / human languages (English, Arabic, French, Spanish, German, ...)
- Academic subjects taken in school (Math, Physics, Chemistry, Religion, Economics, Biology, Arts, ...)
- Generic adjectives or proficiency qualifiers (basic, advanced, good, excellent, fluent, ...)
- Industries, companies, or organisation types (Government, Apple, Agency, Hospital, ...)
- Personal interests / hobbies (painting, sports, fine dining, reading, movies, ...)
- Job titles or roles (Designer, Manager, Engineer, ...)

Rules:
- Use the resume's own wording. Do not invent skills.
- Deduplicate phrases that refer to the same concept.
- Prefer specific phrases over generic ones (e.g., "Adobe Photoshop" over "photo").
- Limit to at most 60 entries.
- Output JSON only. No prose, no markdown fences.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


EXPERIENCE_PROMPT = """You are an experience-extraction agent. Estimate
work-experience features from the resume text. Return ONLY a JSON object
with these keys:

- total_years: number, total years of professional experience as a float.
- role_count: integer, number of distinct paid roles or positions.
- leadership_indicators: integer, count of statements showing leadership (led, managed, mentored, ...).
- impact_statements: integer, count of statements describing measurable impact (improved, reduced, delivered, ...).
- quantified_impact: integer, subset of impact_statements that include a number or percentage.
- evidence: array of up to 3 short verbatim sentences supporting the counts above.

Rules:
- Be conservative; if unsure, return 0.
- Do not extrapolate years beyond what the resume states.
- Output JSON only.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


EVALUATOR_PROMPT = """You are a hiring evaluator. A deterministic scorer has
already produced the numeric fit_score below. Write a concise, calibrated,
evidence-grounded rationale consistent with the score and the signals.

Job description:
\"\"\"
{job_description}
\"\"\"

Required skills: {required_skills}
Preferred skills: {preferred_skills}

Candidate signals:
- canonical skills matched: {matched_skills}
- canonical skills missing: {missing_skills}
- experience years (estimated): {total_years}
- leadership indicators: {leadership}
- text similarity (0..1): {text_similarity}
- fit_score (0..100): {fit_score}

Return ONLY a JSON object with these keys:
- rationale: 2-3 sentence justification. MUST name at least one specific skill
  from `matched_skills` (as evidence of fit) AND at least one specific skill
  from `missing_skills` (as evidence of gap). Prefer concrete skill names over
  generic phrases.
- strengths: array of up to 5 items drawn from `matched_skills` or from the
  listed experience signals (years, leadership, impact). Use the exact
  canonical skill names.
- gaps: array of up to 5 items drawn from `missing_skills`. Use the exact
  canonical skill names.
- recommendation: one short sentence whose tone matches fit_score
  (>=70 = recommend, 40-69 = consider with reservations, <40 = do not advance).

Rules:
- Do NOT invent skills that are not in `matched_skills` or `missing_skills`.
- Do NOT contradict fit_score.
- Do NOT reference demographic attributes (gender, race, age, religion,
  nationality, marital or family status, disability, sexual orientation, names).
- Keep language neutral and professional.
- Output JSON only. No prose, no markdown fences.
"""
