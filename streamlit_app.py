"""
Autonomous Streamlit app: Intent classifier + Session question agents via OpenRouter.
Uploaded JSON is passed to the model as provided (no field validation or stripping).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# llm_language.py (verbatim)
# ---------------------------------------------------------------------------
SYSTEM_LANGUAGE_DIRECTIVE = (
    "Language policy: Use English by default for all visible text you produce. "
    "If the user's latest message (or the user-authored strings in the JSON payload) is clearly "
    "not English, write your entire reply in that same language instead. "
    "When in doubt, prefer English. "
    "CRITICAL TONE RULE: NEVER expose internal system mechanics, logic, or variables to the user. "
    "DO NOT mention technical terms like 'null', 'JSON', 'intents', 'skill_creation', 'session_question', or database IDs. "
    "Communicate as a natural, helpful, and professional human assistant."
)

# ---------------------------------------------------------------------------
# session_question_agent helpers
# ---------------------------------------------------------------------------
_ANSWER_KEYS = ("assistant_answer", "answer",
                "response", "reply", "message", "text")


def _preview_llm_text(text: str | None, limit: int = 600) -> str:
    if not text:
        return ""
    t = text.strip()
    return t if len(t) <= limit else t[:limit] + "…"


def _extract_answer_from_parsed(parsed: Any) -> str:
    """Robustly pull a textual answer out of whatever shape the LLM returned."""
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in _ANSWER_KEYS:
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in parsed.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(parsed, list):
        for item in parsed:
            found = _extract_answer_from_parsed(item)
            if found:
                return found
    return ""


SUPPORTED_INTENTS = frozenset(
    {"skill_creation", "general_chat", "session_question", "visualization_request"})

# Defaults for OpenRouter completion calls (session + intent agents)
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-pro"
DEFAULT_COMPLETION_PERSONALIZATION_MODEL = "openai/gpt-4o-mini"
DEFAULT_SESSION_ENGAGEMENT_MODEL = "google/gemini-2.5-pro"
LLM_TEMPERATURE = 0.1

MODEL_PRESETS: list[str] = [
    DEFAULT_OPENROUTER_MODEL,
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-001",
    "z-ai/glm-5",
    "z-ai/glm-5.1",
    "z-ai/glm-4.7",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2-0905",
    "moonshotai/kimi-k2-thinking",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.6-plus",
]

# OpenRouter `reasoning.effort`; None = omit field (same as disabling extended reasoning).
REASONING_EFFORT_OPTIONS: list[tuple[str, str | None]] = [
    ("None", None),
    ("Low", "low"),
    ("Medium", "medium"),
    ("High", "high"),
]

AGENT_MODEL_DEFAULTS: dict[str, str] = {
    "intent_classifier": DEFAULT_OPENROUTER_MODEL,
    "session_question": DEFAULT_OPENROUTER_MODEL,
    "visualization_agent": DEFAULT_OPENROUTER_MODEL,
    "completion_personalization": DEFAULT_COMPLETION_PERSONALIZATION_MODEL,
    "session_engagement": DEFAULT_SESSION_ENGAGEMENT_MODEL,
}

STARTER_OUTPUT_LANG_OPTIONS: list[tuple[str, str]] = [
    ("English", "english"),
    ("German", "german"),
    ("Ukrainian", "ukrainian"),
    ("Russian", "russian"),
]


def build_session_messages(
    *,
    question: str,
    session_json: Any,
    steps_json: Any,
    recent_history: list[dict[str, str]],
    pending_skill_question: str | None,
) -> list[dict[str, str]]:
    """
    session_json / steps_json: values from json.loads of uploaded files (any JSON value).
    Embedded in the user message; OpenRouter request uses requests.post(..., json=payload).
    """
    session_context = {
        "session": session_json,
        "steps": steps_json,
    }

    rules = [
        "Only answer using information in session_context (full 'session' and 'steps' as uploaded).",
        "If the answer is not present, clearly state that the information is unavailable.",
        "Cite concrete values from the data when answering factual questions.",
        "Keep answers concise, factual, professional.",
        "CRITICAL LANGUAGE RULE: Always write the value of 'assistant_answer' in the same natural language as 'user_question'.",
        "Output format: a single JSON object {\"assistant_answer\": \"...\"}. Do NOT return a JSON array, do NOT wrap the object in a list.",
    ]

    if pending_skill_question:
        rules.append(
            f"CRITICAL: After answering the user's question, you MUST seamlessly transition and ask them the following pending question to resume the skill creation flow: '{pending_skill_question}'"
        )

    prompt = {
        "task": (
            "You are a data-driven assistant analyzing a user's web automation session. "
            "Answer the user's question using ONLY the data in 'session_context'."
        ),
        "rules": rules,
        "session_context": session_context,
        "recent_messages": (recent_history or [])[-6:],
        "user_question": question,
        "output": {"assistant_answer": "your concise answer"},
    }

    return [
        {
            "role": "system",
            "content": f"{SYSTEM_LANGUAGE_DIRECTIVE} Return JSON only. You are an analytical session assistant.",
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


def build_intent_classifier_messages(message: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    prompt = {
        "task": "Classify the user's message intent into exactly one of the four allowed values.",
        "rules": [
            "Output 'visualization_request' if the user asks for a chart, graph, visualization, plot, diagram, "
            "comparison, ranking, distribution, trend, how many times something happened, how much time was spent, "
            "or any request that implies a visual/graphical representation of data.",
            "Output 'session_question' if the user asks a factual question about tasks, actions, goals, "
            "or metrics within the current session or report — but does NOT imply a chart.",
            "Output 'general_chat' if the user is greeting, small-talking, or asking general questions "
            "unrelated to the session data.",
            "Output 'skill_creation' only if the user explicitly asks to create or configure a new skill or automation.",
            "When in doubt between visualization_request and session_question, prefer visualization_request "
            "if any graphical representation is implied.",
        ],
        "recent_messages": history[-6:],
        "user_message": message,
        "output": {"intent": "general_chat | session_question | visualization_request | skill_creation"},
    }
    return [
        {
            "role": "system",
            "content": f"{SYSTEM_LANGUAGE_DIRECTIVE} Return JSON only. You are an intent classifier.",
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# OpenRouter HTTP
# ---------------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def supports_json_format(model: str) -> bool:
    """Prefer json_object; retry without it on 400 from the API."""
    _ = model
    return True


@dataclass
class OpenRouterResult:
    text: str
    status_code: int
    response_json: dict[str, Any] | None


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    *,
    use_json_object: bool,
    pipeline: list[dict[str, Any]],
    agent_name: str,
    reasoning_effort: str | None = "high",
) -> OpenRouterResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/streamlit/streamlit",
        "X-Title": "UAE Chat Agents Demo",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if use_json_object and supports_json_format(model):
        payload["response_format"] = {"type": "json_object"}

    def _post(body: dict[str, Any]) -> Any:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        r = requests.post(OPENROUTER_URL, headers=headers,
                          json=body, timeout=120)
        resp_json: dict[str, Any] | None = None
        try:
            resp_json = r.json()
        except Exception:
            resp_json = None
        raw_text = ""
        if resp_json and isinstance(resp_json, dict):
            choices = resp_json.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(
                    choices[0], dict) else None
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    raw_text = msg["content"]
        if not raw_text and r.text:
            raw_text = r.text
        pipeline.append({
            "agent": agent_name,
            "timestamp_utc": ts,
            "model": model,
            "temperature": temperature,
            "request": {"url": OPENROUTER_URL, "body": json.loads(json.dumps(body))},
            "response": {
                "status_code": r.status_code,
                "raw_text": raw_text[:12000] if len(raw_text) > 12000 else raw_text,
                "parsed_json": resp_json,
            },
        })
        return r, resp_json, raw_text

    r, resp_json, raw_text = _post(payload)
    working = dict(payload)
    if r.status_code == 400 and "response_format" in working:
        working = {k: v for k, v in working.items() if k != "response_format"}
        r, resp_json, raw_text = _post(working)
    if r.status_code == 400 and "reasoning" in working:
        working = {k: v for k, v in working.items() if k != "reasoning"}
        r, resp_json, raw_text = _post(working)
    if r.status_code == 400 and "response_format" in working:
        working = {k: v for k, v in working.items() if k != "response_format"}
        r, resp_json, raw_text = _post(working)
    if r.status_code == 400:
        minimal = {k: v for k, v in payload.items() if k not in (
            "response_format", "reasoning")}
        r, resp_json, raw_text = _post(minimal)

    return OpenRouterResult(text=raw_text, status_code=r.status_code, response_json=resp_json)


def parse_session_answer(raw_text: str) -> str:
    answer = ""
    try:
        parsed = json.loads(raw_text)
        answer = _extract_answer_from_parsed(parsed)
    except json.JSONDecodeError:
        answer = raw_text.strip()
    except Exception:
        answer = raw_text.strip()

    if not answer:
        answer = "I could not generate an answer from the session data."
    return answer


def parse_intent(raw_text: str) -> str | None:
    try:
        parsed = json.loads(raw_text)
        intent = str(parsed.get("intent") or "").strip().lower()
        if intent in {"session_question", "general_chat", "visualization_request", "skill_creation"}:
            return intent
    except Exception:
        pass
    return None


def extract_first_complete_completion_rationale(session_json: Any) -> str | None:
    """
    From uploaded session JSON, return completion_rationale for the first goal entry
    (in list order under the first discovered `goals` array) where goal_completion_status is 'complete'.
    """

    def collect_goal_lists(node: Any) -> list[list[Any]]:
        found: list[list[Any]] = []
        if isinstance(node, dict):
            g = node.get("goals")
            if isinstance(g, list):
                found.append(g)
            for v in node.values():
                found.extend(collect_goal_lists(v))
        elif isinstance(node, list):
            for item in node:
                found.extend(collect_goal_lists(item))
        return found

    for goals in collect_goal_lists(session_json):
        for item in goals:
            if not isinstance(item, dict):
                continue
            status = str(item.get("goal_completion_status")
                         or "").strip().lower()
            if status != "complete":
                continue
            cr = item.get("completion_rationale")
            if isinstance(cr, str) and cr.strip():
                return cr.strip()
    return None


def parse_json_single_text(raw_text: str, primary_key: str) -> str:
    """Parse model JSON output; fall back to whole body or generic answer keys."""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            v = parsed.get(primary_key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            return _extract_answer_from_parsed(parsed)
    except json.JSONDecodeError:
        return (raw_text or "").strip()
    except Exception:
        return (raw_text or "").strip()
    return (raw_text or "").strip()


def build_completion_personalization_messages(
    *,
    user_name: str,
    output_language: str,
    completion_rationale: str,
) -> list[dict[str, str]]:
    prompt = {
        "task": (
            "Rewrite the session completion rationale into one short, warm, personalized line for the user. "
            "The very first words must be a casual hello that uses their name, like \"Hey Peter, …\" in English."
        ),
        "inputs": {
            "user_name": user_name,
            "output_language": output_language,
            "completion_rationale": completion_rationale,
        },
        "rules": [
            "CRITICAL: Start personalized_message with a short informal salutation that includes user_name as the addressee, "
            "in the same spirit as English \"Hey Peter,\" or \"Hi Anna,\" — pick the natural equivalent for output_language (not Dear Sir/Madam).",
            "Convey that they have just finished a task or goal, using only facts implied by completion_rationale; do not invent metrics or events.",
            "Write the entire message in the language indicated by output_language: "
            "english, german, ukrainian, or russian.",
            "Do not mention JSON, schemas, fields, tiers, models, or any system internals.",
            "Do not add explanations, labels, or markdown; the user sees only the final line.",
            'Output format: a single JSON object {"personalized_message": "..."}. '
            "Do not return an array; do not wrap the object in a list.",
        ],
    }
    return [
        {
            "role": "system",
            "content": "Return JSON only. You personalize brief user-facing completion messages.",
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


def build_session_engagement_messages(
    *,
    output_language: str,
    session_json: Any,
    steps_json: Any,
) -> list[dict[str, str]]:
    session_context = {"session": session_json, "steps": steps_json}
    prompt = {
        "task": (
            "From the session data, produce one engaging message that highlights interesting, data-backed facts "
            "(e.g. apps visited, repetition, flow between tools) and invites the reader to chat more about the session."
        ),
        "inputs": {
            "output_language": output_language,
            "session_context": session_context,
        },
        "rules": [
            "CRITICAL: Do not greet. No salutations or openers such as Hi, Hello, Hey, Dear, Good morning/afternoon, "
            "or any equivalent in the output language (including non-English). Start directly with the fact or hook.",
            "Use a hook such as 'Did you know ...?' or a rhetorical question — without any greeting before it.",
            "Ground every claim in session_context; if something is not in the data, do not assert it.",
            "If the data is sparse, still offer a light, curiosity-driven prompt tied to what is known.",
            "Write entirely in the language indicated by output_language: english, german, ukrainian, or russian.",
            "Encourage further conversation about the session data (e.g. offer to explore more).",
            "Do not mention JSON, schemas, uploads, or system internals.",
            "Do not add explanations or headings; output only the user-facing line inside the JSON value.",
            'Output format: a single JSON object {"engagement_message": "..."}. '
            "Do not return an array; do not wrap the object in a list.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. You surface engaging, accurate insights from session analytics. "
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


# ---------------------------------------------------------------------------
# Visualization: 14 chart-spec schemas + Plotly dark-theme renderer
# ---------------------------------------------------------------------------

CHART_SPEC_SCHEMAS: dict[str, Any] = {
    "bar": {
        "chart_type": "bar",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["str | number — category labels"],
        "y_values": ["number — bar heights, same length as x_values"],
    },
    "horizontal_bar": {
        "chart_type": "horizontal_bar",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["number — bar lengths, same length as y_values"],
        "y_values": ["str — category labels"],
    },
    "line": {
        "chart_type": "line",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["str | number — step or time labels"],
        "y_values": ["number — same length as x_values"],
    },
    "area": {
        "chart_type": "area",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["str | number"],
        "y_values": ["number — same length as x_values"],
    },
    "pie": {
        "chart_type": "pie",
        "title": "str",
        "labels": ["str — slice names"],
        "values": ["number — slice sizes, same length as labels"],
    },
    "donut": {
        "chart_type": "donut",
        "title": "str",
        "center_label": "str — text shown in the hole",
        "labels": ["str"],
        "values": ["number — same length as labels"],
    },
    "scatter": {
        "chart_type": "scatter",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["number"],
        "y_values": ["number — same length as x_values"],
        "point_labels": ["str — one per point, same length as x_values"],
    },
    "stacked_bar": {
        "chart_type": "stacked_bar",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["str — category labels"],
        "series": [{"name": "str", "values": ["number — one per x_value"]}],
    },
    "funnel": {
        "chart_type": "funnel",
        "title": "str",
        "stages": ["str — stage names ordered wide to narrow"],
        "values": ["number — same length as stages"],
    },
    "gauge": {
        "chart_type": "gauge",
        "title": "str",
        "value": "number",
        "min_value": "number",
        "max_value": "number",
        "label": "str — unit or short description",
    },
    "bubble": {
        "chart_type": "bubble",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["number"],
        "y_values": ["number — same length as x_values"],
        "sizes": ["number — bubble sizes, same length as x_values"],
        "labels": ["str — point labels, same length as x_values"],
    },
    "treemap": {
        "chart_type": "treemap",
        "title": "str",
        "labels": ["str — node names (include parent names too)"],
        "parents": ["str — parent name; empty string for root nodes"],
        "values": ["number — node sizes, same length as labels"],
    },
    "heatmap": {
        "chart_type": "heatmap",
        "title": "str",
        "x_label": "str", "y_label": "str",
        "x_values": ["str — column labels"],
        "y_values": ["str — row labels"],
        "z_values": [["number — row×col matrix; rows==len(y_values), cols==len(x_values)"]],
    },
    "timeline": {
        "chart_type": "timeline",
        "title": "str",
        "tasks": ["str — task or goal names"],
        "start_values": ["number — start step id, same length as tasks"],
        "end_values": ["number — end step id, same length as tasks"],
        "categories": ["str — grouping label per task, same length as tasks"],
    },
}

_CHART_TYPE_GUIDANCE: dict[str, str] = {
    "bar":           "counts, scores, or comparisons across ≤12 categories (vertical bars)",
    "horizontal_bar": "rankings or comparisons with long category names (horizontal bars)",
    "line":          "trends or progression over sequential steps or time (connected points)",
    "area":          "cumulative trends or time-on-task (filled line)",
    "pie":           "proportions of a whole — best with 2–7 slices",
    "donut":         "proportions with a center summary label or total",
    "scatter":       "correlation between two numeric metrics, one point per goal/action",
    "stacked_bar":   "breakdown per category into sub-groups (2+ stacked series)",
    "funnel":        "pipeline stages, completion drop-off, or sequential narrowing",
    "gauge":         "single KPI or percentage — exactly one numeric value with min/max",
    "bubble":        "three-dimensional view: x-position, y-position, and a size dimension",
    "treemap":       "hierarchical proportions — tool → goal → sub-action sizes",
    "heatmap":       "frequency or intensity matrix across two categorical axes",
    "timeline":      "step-id or time ranges per goal/task (Gantt-style bars)",
}

# Plotly dark-theme palette & helpers
_VIZ_COLORS = [
    "#818cf8", "#a78bfa", "#f472b6", "#fb7185", "#fb923c",
    "#fbbf24", "#34d399", "#22d3ee", "#38bdf8", "#60a5fa",
]
_VIZ_DARK_BG = "#0f172a"
_VIZ_PAPER = "#1e293b"
_VIZ_TEXT = "#e2e8f0"
_VIZ_GRID = "#334155"
_VIZ_AXIS = "#64748b"


def _apply_dark_layout(
    fig: Any,
    title: str = "",
    margin: dict[str, int] | None = None,
) -> Any:
    m = margin or {"t": 55, "b": 50, "l": 65, "r": 20, "pad": 4}
    fig.update_layout(
        title={"text": title, "font": {
            "size": 16, "color": _VIZ_TEXT}, "x": 0.01},
        paper_bgcolor=_VIZ_PAPER,
        plot_bgcolor=_VIZ_DARK_BG,
        font={"family": "Inter, system-ui, sans-serif", "color": _VIZ_TEXT},
        margin=m,
        colorway=_VIZ_COLORS,
        legend={
            "bgcolor": "rgba(15,23,42,0.8)", "bordercolor": _VIZ_GRID,
            "borderwidth": 1, "font": {"size": 12},
        },
        hoverlabel={
            "bgcolor": "#1e293b", "bordercolor": "#475569",
            "font": {"color": _VIZ_TEXT},
        },
    )
    fig.update_xaxes(
        gridcolor=_VIZ_GRID, linecolor=_VIZ_AXIS,
        tickfont={"color": _VIZ_TEXT}, title_font={"color": _VIZ_TEXT},
        showgrid=True, zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=_VIZ_GRID, linecolor=_VIZ_AXIS,
        tickfont={"color": _VIZ_TEXT}, title_font={"color": _VIZ_TEXT},
        showgrid=True, zeroline=False,
    )
    return fig


def build_visualization_messages(
    *,
    question: str,
    session_json: Any,
    steps_json: Any,
    recent_history: list[dict[str, str]],
    pending_skill_question: str | None,
) -> list[dict[str, str]]:
    """
    Single call that returns both a text answer and a chart_spec derived from session data.
    The model picks the best chart_type from CHART_SPEC_SCHEMAS and populates it.
    chart_spec is null when the data does not support a meaningful chart.
    """
    session_context = {"session": session_json, "steps": steps_json}
    rules = [
        "Answer the user's question using ONLY the data in 'session_context'.",
        "CRITICAL LANGUAGE RULE: Always write 'assistant_answer' in the same natural language as 'user_question'.",
        "Simultaneously produce a chart_spec object that best visualises the answer.",
        "Step 1 — choose the most insightful chart_type from chart_type_guidance based on user_question and data.",
        "Step 2 — populate chart_spec following EXACTLY the corresponding schema from chart_spec_schemas.",
        "Derive ALL numeric values PURELY from session_context; do NOT invent any numbers.",
        "If session data does not contain enough numeric information for a meaningful chart, set chart_spec to null.",
        "Do not mention JSON, schemas, or system internals in assistant_answer.",
        "Output format: a single JSON object with exactly two keys: 'assistant_answer' (string) and "
        "'chart_spec' (object matching one schema from chart_spec_schemas, or null). "
        "Do NOT return an array; do NOT wrap the object in a list.",
    ]
    if pending_skill_question:
        rules.append(
            f"CRITICAL: After answering, seamlessly ask the following pending question: '{pending_skill_question}'"
        )
    prompt = {
        "task": (
            "You are a data-driven assistant. Answer the user's question AND produce the most insightful chart "
            "that visually represents the answer, both derived strictly from 'session_context'."
        ),
        "rules": rules,
        "chart_type_guidance": _CHART_TYPE_GUIDANCE,
        "chart_spec_schemas": CHART_SPEC_SCHEMAS,
        "session_context": session_context,
        "recent_messages": (recent_history or [])[-6:],
        "user_question": question,
        "output": {
            "assistant_answer": "concise text answer",
            "chart_spec": "< object matching exactly one schema from chart_spec_schemas, or null >",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_LANGUAGE_DIRECTIVE} Return JSON only. "
                "You are an analytical session assistant that produces both text answers and chart specs."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]


def _validate_chart_spec(cs: dict[str, Any]) -> bool:
    """Return True if cs has the minimum required fields for its chart_type."""
    ct = str(cs.get("chart_type", "")).lower()
    if ct in ("bar", "line", "area", "horizontal_bar"):
        x, y = cs.get("x_values"), cs.get("y_values")
        return isinstance(x, list) and isinstance(y, list) and 0 < len(x) == len(y)
    if ct in ("pie", "donut"):
        lbl, val = cs.get("labels"), cs.get("values")
        return isinstance(lbl, list) and isinstance(val, list) and 0 < len(lbl) == len(val)
    if ct == "scatter":
        x, y = cs.get("x_values"), cs.get("y_values")
        ok = isinstance(x, list) and isinstance(
            y, list) and 0 < len(x) == len(y)
        if ok and cs.get("point_labels") is not None:
            ok = isinstance(cs["point_labels"], list) and len(
                cs["point_labels"]) == len(x)
        return ok
    if ct == "stacked_bar":
        x, series = cs.get("x_values"), cs.get("series")
        if not isinstance(x, list) or not x or not isinstance(series, list) or not series:
            return False
        return all(
            isinstance(s, dict) and isinstance(
                s.get("values"), list) and len(s["values"]) == len(x)
            for s in series
        )
    if ct == "funnel":
        s, v = cs.get("stages"), cs.get("values")
        return isinstance(s, list) and isinstance(v, list) and 0 < len(s) == len(v)
    if ct == "gauge":
        return isinstance(cs.get("value"), (int, float))
    if ct == "bubble":
        x, y, sz = cs.get("x_values"), cs.get("y_values"), cs.get("sizes")
        return (isinstance(x, list) and isinstance(y, list) and isinstance(sz, list)
                and 0 < len(x) == len(y) == len(sz))
    if ct == "treemap":
        lbl, par, val = cs.get("labels"), cs.get("parents"), cs.get("values")
        return (isinstance(lbl, list) and isinstance(par, list) and isinstance(val, list)
                and 0 < len(lbl) == len(par) == len(val))
    if ct == "heatmap":
        x, y, z = cs.get("x_values"), cs.get("y_values"), cs.get("z_values")
        return (isinstance(x, list) and isinstance(y, list) and isinstance(z, list)
                and len(z) == len(y) > 0
                and all(isinstance(row, list) and len(row) == len(x) for row in z))
    if ct == "timeline":
        t, sv, ev = cs.get("tasks"), cs.get(
            "start_values"), cs.get("end_values")
        return (isinstance(t, list) and isinstance(sv, list) and isinstance(ev, list)
                and 0 < len(t) == len(sv) == len(ev))
    return False


def parse_visualization_response(raw_text: str) -> tuple[str, dict[str, Any] | None]:
    """Return (assistant_answer, chart_spec_or_None)."""
    answer = ""
    chart_spec: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            answer = str(parsed.get("assistant_answer") or "").strip()
            cs = parsed.get("chart_spec")
            if isinstance(cs, dict) and cs and _validate_chart_spec(cs):
                chart_spec = cs
    except Exception:
        answer = raw_text.strip()
    if not answer:
        answer = "I could not generate an answer from the session data."
    return answer, chart_spec


def render_chart_spec(spec: dict[str, Any]) -> None:
    """Render a chart_spec with Plotly (dark modern theme). 14 chart types supported."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly to enable charts: `pip install plotly`")
        return

    ct = str(spec.get("chart_type", "bar")).lower()
    title = str(spec.get("title", ""))
    st.markdown(f"**chart_type:** `{ct}`")

    def _colors(n: int) -> list[str]:
        return (_VIZ_COLORS * ((n // len(_VIZ_COLORS)) + 1))[:n]

    # ── bar / horizontal_bar ─────────────────────────────────────────────────
    if ct in ("bar", "horizontal_bar"):
        x_vals = spec.get("x_values", [])
        y_vals = spec.get("y_values", [])
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        clrs = _colors(max(len(x_vals), len(y_vals)))
        if ct == "horizontal_bar":
            fig = go.Figure(go.Bar(
                x=x_vals, y=y_vals, orientation="h",
                marker=dict(color=clrs, line=dict(width=0)),
                hovertemplate=f"<b>%{{y}}</b><br>{xl}: %{{x}}<extra></extra>",
            ))
            fig.update_xaxes(title_text=xl)
            fig.update_yaxes(title_text=yl, autorange="reversed")
        else:
            fig = go.Figure(go.Bar(
                x=x_vals, y=y_vals,
                marker=dict(color=clrs, line=dict(width=0)),
                hovertemplate=f"<b>%{{x}}</b><br>{yl}: %{{y}}<extra></extra>",
            ))
            fig.update_xaxes(title_text=xl)
            fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── line / area ──────────────────────────────────────────────────────────
    elif ct in ("line", "area"):
        x_vals = spec.get("x_values", [])
        y_vals = spec.get("y_values", [])
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        fig = go.Figure(go.Scatter(
            x=x_vals, y=y_vals, mode="lines+markers",
            line=dict(color=_VIZ_COLORS[0], width=2.5, shape="spline"),
            marker=dict(size=7, color=_VIZ_COLORS[0], line=dict(
                color=_VIZ_PAPER, width=1.5)),
            fill="tozeroy" if ct == "area" else None,
            fillcolor="rgba(129,140,248,0.15)" if ct == "area" else None,
            hovertemplate=f"<b>%{{x}}</b><br>{yl}: %{{y}}<extra></extra>",
        ))
        fig.update_xaxes(title_text=xl)
        fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── pie / donut ──────────────────────────────────────────────────────────
    elif ct in ("pie", "donut"):
        labels = spec.get("labels", [])
        values = spec.get("values", [])
        clrs = _colors(len(labels))
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.45 if ct == "donut" else 0,
            marker=dict(colors=clrs, line=dict(color=_VIZ_DARK_BG, width=2)),
            textfont=dict(color=_VIZ_TEXT),
            hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
        ))
        if ct == "donut":
            center = str(spec.get("center_label", ""))
            if center:
                fig.update_layout(annotations=[dict(
                    text=center, x=0.5, y=0.5, showarrow=False,
                    font=dict(size=15, color=_VIZ_TEXT),
                )])
        fig.update_layout(
            paper_bgcolor=_VIZ_PAPER, plot_bgcolor=_VIZ_DARK_BG,
            font={"family": "Inter, system-ui, sans-serif", "color": _VIZ_TEXT},
            title={"text": title, "font": {
                "size": 16, "color": _VIZ_TEXT}, "x": 0.01},
            margin={"t": 55, "b": 20, "l": 20, "r": 20},
            legend={"font": {"color": _VIZ_TEXT}, "bgcolor": "rgba(0,0,0,0)"},
            colorway=_VIZ_COLORS,
            hoverlabel={"bgcolor": "#1e293b", "bordercolor": "#475569",
                        "font": {"color": _VIZ_TEXT}},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── scatter ──────────────────────────────────────────────────────────────
    elif ct == "scatter":
        x_vals = spec.get("x_values", [])
        y_vals = spec.get("y_values", [])
        pt_labels = spec.get("point_labels") or [""] * len(x_vals)
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        fig = go.Figure(go.Scatter(
            x=x_vals, y=y_vals, mode="markers+text",
            text=pt_labels, textposition="top center",
            textfont=dict(color=_VIZ_TEXT, size=10),
            marker=dict(size=12, color=_VIZ_COLORS[0],
                        line=dict(color=_VIZ_PAPER, width=1.5)),
            hovertemplate=f"<b>%{{text}}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
        ))
        fig.update_xaxes(title_text=xl)
        fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── stacked_bar ──────────────────────────────────────────────────────────
    elif ct == "stacked_bar":
        x_vals = spec.get("x_values", [])
        series = spec.get("series", [])
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        fig = go.Figure()
        for i, s in enumerate(series):
            fig.add_trace(go.Bar(
                name=s.get("name", f"Series {i + 1}"),
                x=x_vals, y=s.get("values", []),
                marker=dict(color=_VIZ_COLORS[i % len(
                    _VIZ_COLORS)], line=dict(width=0)),
                hovertemplate=f"<b>%{{x}}</b><br>{s.get('name', '')}: %{{y}}<extra></extra>",
            ))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text=xl)
        fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── funnel ───────────────────────────────────────────────────────────────
    elif ct == "funnel":
        stages = spec.get("stages", [])
        values = spec.get("values", [])
        fig = go.Figure(go.Funnel(
            y=stages, x=values,
            textinfo="value+percent initial",
            marker=dict(color=_colors(len(stages)),
                        line=dict(width=2, color=_VIZ_PAPER)),
            connector=dict(line=dict(color=_VIZ_GRID, width=1)),
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=_VIZ_PAPER, plot_bgcolor=_VIZ_DARK_BG,
            font={"family": "Inter, system-ui, sans-serif", "color": _VIZ_TEXT},
            title={"text": title, "font": {
                "size": 16, "color": _VIZ_TEXT}, "x": 0.01},
            margin={"t": 55, "b": 30, "l": 130, "r": 20},
            hoverlabel={"bgcolor": "#1e293b", "bordercolor": "#475569",
                        "font": {"color": _VIZ_TEXT}},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── gauge ────────────────────────────────────────────────────────────────
    elif ct == "gauge":
        value = float(spec.get("value", 0))
        min_val = float(spec.get("min_value", 0))
        max_val = float(spec.get("max_value", 100))
        label = str(spec.get("label", ""))
        pct = (value - min_val) / (max_val -
                                   min_val) if max_val > min_val else 0
        bar_clr = _VIZ_COLORS[6] if pct >= 0.7 else (
            _VIZ_COLORS[4] if pct >= 0.4 else _VIZ_COLORS[3])
        span = max_val - min_val
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"color": _VIZ_TEXT, "size": 40}},
            title={"text": label, "font": {"color": _VIZ_TEXT, "size": 14}},
            gauge={
                "axis": {"range": [min_val, max_val],
                         "tickcolor": _VIZ_TEXT, "tickfont": {"color": _VIZ_TEXT}},
                "bar": {"color": bar_clr, "thickness": 0.25},
                "bgcolor": _VIZ_DARK_BG,
                "borderwidth": 0,
                "steps": [
                    {"range": [min_val, min_val + span * 0.4],
                     "color": "rgba(100,116,139,0.15)"},
                    {"range": [min_val + span * 0.4, min_val + span * 0.7],
                     "color": "rgba(100,116,139,0.3)"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor=_VIZ_PAPER,
            font={"family": "Inter, system-ui, sans-serif", "color": _VIZ_TEXT},
            title={"text": title, "font": {
                "size": 16, "color": _VIZ_TEXT}, "x": 0.01},
            margin={"t": 55, "b": 30, "l": 30, "r": 30},
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── bubble ───────────────────────────────────────────────────────────────
    elif ct == "bubble":
        x_vals = spec.get("x_values", [])
        y_vals = spec.get("y_values", [])
        sizes = spec.get("sizes", [])
        labels = spec.get("labels") or [""] * len(x_vals)
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        max_sz = max((float(s) for s in sizes), default=1) or 1
        norm = [max(8, 60 * float(s) / max_sz) for s in sizes]
        clrs = _colors(len(x_vals))
        fig = go.Figure(go.Scatter(
            x=x_vals, y=y_vals, mode="markers+text",
            text=labels, textposition="top center",
            textfont=dict(color=_VIZ_TEXT, size=10),
            marker=dict(size=norm, color=clrs, opacity=0.8,
                        line=dict(color=_VIZ_PAPER, width=1)),
            hovertemplate=f"<b>%{{text}}</b><br>{xl}: %{{x}}<br>{yl}: %{{y}}<extra></extra>",
        ))
        fig.update_xaxes(title_text=xl)
        fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── treemap ──────────────────────────────────────────────────────────────
    elif ct == "treemap":
        labels = spec.get("labels", [])
        parents = spec.get("parents", [])
        values = spec.get("values", [])
        n = len(labels)
        cscale = [[i / max(n - 1, 1), _VIZ_COLORS[i % len(_VIZ_COLORS)]]
                  for i in range(n)]
        fig = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=list(range(n)), colorscale=cscale,
                        line=dict(width=1, color=_VIZ_PAPER)),
            textfont=dict(color=_VIZ_TEXT, size=12),
            hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=_VIZ_PAPER,
            font={"family": "Inter, system-ui, sans-serif", "color": _VIZ_TEXT},
            title={"text": title, "font": {
                "size": 16, "color": _VIZ_TEXT}, "x": 0.01},
            margin={"t": 55, "b": 10, "l": 10, "r": 10},
            hoverlabel={"bgcolor": "#1e293b", "bordercolor": "#475569",
                        "font": {"color": _VIZ_TEXT}},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── heatmap ──────────────────────────────────────────────────────────────
    elif ct == "heatmap":
        x_vals = spec.get("x_values", [])
        y_vals = spec.get("y_values", [])
        z_vals = spec.get("z_values", [[]])
        xl = str(spec.get("x_label", ""))
        yl = str(spec.get("y_label", ""))
        fig = go.Figure(go.Heatmap(
            x=x_vals, y=y_vals, z=z_vals,
            colorscale=[
                [0.0, _VIZ_DARK_BG],
                [0.5, _VIZ_COLORS[0]],
                [1.0, _VIZ_COLORS[2]],
            ],
            showscale=True,
            hovertemplate=f"{xl}: %{{x}}<br>{yl}: %{{y}}<br>Value: %{{z}}<extra></extra>",
        ))
        fig.update_xaxes(title_text=xl)
        fig.update_yaxes(title_text=yl)
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)

    # ── timeline (Gantt) ─────────────────────────────────────────────────────
    elif ct == "timeline":
        tasks = spec.get("tasks", [])
        starts = spec.get("start_values", [])
        ends = spec.get("end_values", [])
        cats = spec.get("categories") or [""] * len(tasks)
        unique = list(dict.fromkeys(cats))
        fig = go.Figure()
        seen: set[str] = set()
        for task, s, e, cat in zip(tasks, starts, ends, cats):
            ci = unique.index(cat) if cat else tasks.index(task)
            color = _VIZ_COLORS[ci % len(_VIZ_COLORS)]
            dur = max(float(e) - float(s), 0)
            fig.add_trace(go.Bar(
                name=cat or task,
                x=[dur], y=[task],
                base=[float(s)],
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{task}</b><br>Start: {s}<br>End: {e}<br>Duration: {dur}<extra></extra>"
                ),
                showlegend=bool(cat) and cat not in seen,
                legendgroup=cat,
            ))
            if cat:
                seen.add(cat)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="Step / Time")
        fig.update_yaxes(autorange="reversed")
        _apply_dark_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)


def get_api_key() -> str:
    try:
        return str(st.secrets.get("OPENROUTER_API_KEY", "") or "").strip()
    except Exception:
        return ""


def messages_to_agent_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Use chat transcript as conversation_history for agents (role + content)."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _format_ts_utc_caption(ts: str | None) -> str | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        parsed = parsed.astimezone(dt.timezone.utc)
        return parsed.strftime("%H:%M:%S UTC")
    except ValueError:
        return ts.strip()


def chat_timing_maxima(messages: list[dict[str, Any]]) -> tuple[float, float]:
    """Denominators for progress bars: max gap after previous msg, max assistant latency."""
    max_gap = 1.0
    max_lat = 1.0
    for m in messages:
        g = m.get("gap_since_prev_sec")
        if isinstance(g, (int, float)) and g >= 0 and float(g) > max_gap:
            max_gap = float(g)
        rt = m.get("response_time_sec")
        if isinstance(rt, (int, float)) and rt >= 0 and float(rt) > max_lat:
            max_lat = float(rt)
    return max_gap, max_lat


def _welcome_timing_caption(meta: dict[str, Any] | None) -> str | None:
    if not meta or not isinstance(meta, dict):
        return None
    lat = meta.get("latency_sec")
    ts_disp = _format_ts_utc_caption(meta.get("ts_utc"))
    http = meta.get("http_status")
    parts: list[str] = []
    if ts_disp:
        parts.append(f"Completed at {ts_disp}")
    if isinstance(lat, (int, float)) and lat >= 0:
        parts.append(f"Request time: {float(lat):.2f}s")
    if http is not None:
        parts.append(f"HTTP {http}")
    return " · ".join(parts) if parts else None


def render_welcome_flow_blocks() -> None:
    """Welcome messages from Start (not in chat messages): labels + timing captions."""
    sp0 = st.session_state.get("starter_personalized")
    se0 = st.session_state.get("starter_engagement")
    timing: dict[str, Any] = st.session_state.get("starter_welcome_timing") or {}
    meta_p = timing.get("completion_personalization")
    meta_e = timing.get("session_engagement")
    show_p = bool(sp0) or isinstance(meta_p, dict)
    show_e = bool(se0) or isinstance(meta_e, dict)
    if not show_p and not show_e:
        return
    st.subheader("Welcome flow (from Start)")
    if show_p:
        st.markdown("**Completion personalization**")
        if sp0:
            st.markdown(sp0)
        elif isinstance(meta_p, dict) and meta_p.get("http_status") == 200:
            st.caption(
                "No text parsed (`personalized_message`). Inspect the "
                "`completion_personalization` step in the pipeline export."
            )
        if isinstance(meta_p, dict):
            cap = _welcome_timing_caption(meta_p)
            if cap:
                st.caption(cap)
    if show_e:
        st.markdown("**Session engagement**")
        if se0:
            st.markdown(se0)
        elif isinstance(meta_e, dict) and meta_e.get("http_status") == 200:
            st.caption(
                "No text was parsed (expected JSON key `engagement_message`). "
                "See `session_engagement` in the pipeline export."
            )
        if isinstance(meta_e, dict):
            cap_e = _welcome_timing_caption(meta_e)
            if cap_e:
                st.caption(cap_e)
    st.divider()


def render_chat_message_timing(
    m: dict[str, Any], *, max_gap: float, max_latency: float
) -> None:
    """Captions + progress bars for user (send time, gap since previous) and assistant (latency)."""
    role = m.get("role")
    ts_disp = _format_ts_utc_caption(m.get("ts_utc"))

    if role == "user":
        cap_parts: list[str] = []
        if ts_disp:
            cap_parts.append(f"Sent at {ts_disp}")
        g = m.get("gap_since_prev_sec")
        if isinstance(g, (int, float)) and g >= 0:
            cap_parts.append(f"Since previous message: {float(g):.2f}s")
        if cap_parts:
            st.caption(" · ".join(cap_parts))
        if isinstance(g, (int, float)) and g >= 0 and max_gap > 0:
            st.progress(min(1.0, float(g) / max_gap))
        return

    if role == "assistant":
        rt = m.get("response_time_sec")
        cap_parts = []
        if ts_disp:
            cap_parts.append(f"Completed at {ts_disp}")
        if isinstance(rt, (int, float)) and rt >= 0:
            cap_parts.append(f"Response latency: {float(rt):.2f}s")
        if cap_parts:
            st.caption(" · ".join(cap_parts))
        if isinstance(rt, (int, float)) and rt >= 0 and max_latency > 0:
            st.progress(min(1.0, float(rt) / max_latency))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []
    if "upload_session" not in st.session_state:
        st.session_state.upload_session = None
    if "upload_steps" not in st.session_state:
        st.session_state.upload_steps = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "starter_personalized" not in st.session_state:
        st.session_state.starter_personalized = None
    if "starter_engagement" not in st.session_state:
        st.session_state.starter_engagement = None
    if "starter_welcome_timing" not in st.session_state:
        st.session_state.starter_welcome_timing = {}
    if "starter_flow_notice" not in st.session_state:
        st.session_state.starter_flow_notice = ""


def _default_model_preset_index(default_model: str) -> int:
    if default_model in MODEL_PRESETS:
        return 1 + MODEL_PRESETS.index(default_model)
    return 0


def sidebar_agent_openrouter_config() -> dict[str, tuple[str, str | None]]:
    """Sidebar widgets: model + reasoning effort per agent. Returns agent_key -> (model_id, effort_or_none)."""
    cfg: dict[str, tuple[str, str | None]] = {}
    agent_rows: tuple[tuple[str, str], ...] = (
        ("intent_classifier", "Intent classifier"),
        ("session_question", "Session / QA agent"),
        ("visualization_agent", "Visualization agent"),
        ("completion_personalization", "Welcome — completion personalization"),
        ("session_engagement", "Welcome — session engagement"),
    )
    with st.expander("Per-agent model & reasoning", expanded=False):
        for i, (agent_key, title) in enumerate(agent_rows):
            st.markdown(f"**{title}**")
            default_m = AGENT_MODEL_DEFAULTS[agent_key]
            preset_choices = ["Custom"] + MODEL_PRESETS
            choice = st.selectbox(
                "Model",
                preset_choices,
                index=_default_model_preset_index(default_m),
                key=f"agent_model_preset_{agent_key}",
            )
            if choice == "Custom":
                model_v = (
                    st.text_input(
                        "Custom model id",
                        value=default_m,
                        key=f"agent_model_custom_{agent_key}",
                    ).strip()
                    or default_m
                )
            else:
                model_v = choice
            eff_labels = [t[0] for t in REASONING_EFFORT_OPTIONS]
            eff_values = [t[1] for t in REASONING_EFFORT_OPTIONS]
            eff_i = st.selectbox(
                "Reasoning effort",
                list(range(len(REASONING_EFFORT_OPTIONS))),
                index=3,
                format_func=lambda i, labels=eff_labels: labels[int(i)],
                key=f"agent_reasoning_idx_{agent_key}",
            )
            reasoning_v = eff_values[int(eff_i)]
            cfg[agent_key] = (model_v, reasoning_v)
            if i < len(agent_rows) - 1:
                st.divider()
    return cfg


def main() -> None:
    st.set_page_config(page_title="UAE Chat Agents",
                       page_icon="💬", layout="wide")
    init_state()

    api_key = get_api_key()

    with st.sidebar:
        st.header("Settings")
        if not api_key:
            st.warning(
                "Set `OPENROUTER_API_KEY` in Streamlit secrets (Deploy → Secrets) or "
                "create `.streamlit/secrets.toml` locally."
            )
        else:
            st.success("API key loaded from secrets.")

        agent_cfg = sidebar_agent_openrouter_config()

        explicit_intent = st.radio(
            "Explicit intent (routing)",
            sorted(SUPPORTED_INTENTS),
            index=sorted(SUPPORTED_INTENTS).index("general_chat"),
            help="Mirrors API explicit_intent: general_chat runs LLM classifier; session_question skips to session agent.",
        )

        st.subheader("Session data (JSON uploads)")
        f_sess = st.file_uploader(
            "Session (JSON) — any structure; sent as `session` in the model context",
            type=["json"],
            key="session_json",
        )
        f_steps = st.file_uploader(
            "Steps (JSON) — any structure; sent as `steps` in the model context",
            type=["json"],
            key="steps_json",
        )

        if f_sess is not None:
            try:
                st.session_state.upload_session = json.loads(
                    f_sess.getvalue().decode("utf-8"))
            except json.JSONDecodeError as e:
                st.error(f"Session file is not valid JSON: {e}")

        if f_steps is not None:
            try:
                st.session_state.upload_steps = json.loads(
                    f_steps.getvalue().decode("utf-8"))
            except json.JSONDecodeError as e:
                st.error(f"Steps file is not valid JSON: {e}")

        st.subheader("Welcome agents (pre-chat)")
        starter_user_name = st.text_input(
            "User name", value="Peter", key="starter_user_name")
        lang_pair = st.selectbox(
            "Output language",
            options=STARTER_OUTPUT_LANG_OPTIONS,
            index=0,
            format_func=lambda pair: pair[0],
            key="starter_lang_select",
        )
        output_language_code = lang_pair[1]

        if st.button("Start", key="starter_run_button"):
            if not api_key:
                st.error("Configure OPENROUTER_API_KEY first.")
            else:
                sess = st.session_state.get("upload_session")
                steps_pl = st.session_state.get("upload_steps")
                rationale = extract_first_complete_completion_rationale(
                    sess) if sess is not None else None
                if sess is None:
                    st.error("Upload a Session (JSON) file first.")
                elif not rationale:
                    st.error(
                        "No completion_rationale found for a goal with goal_completion_status "
                        "'complete' in the session JSON."
                    )
                else:
                    st.session_state.starter_flow_notice = ""
                    st.session_state.starter_welcome_timing = {}
                    st.session_state.starter_personalized = None
                    st.session_state.starter_engagement = None
                    m1 = build_completion_personalization_messages(
                        user_name=starter_user_name.strip() or "Peter",
                        output_language=output_language_code,
                        completion_rationale=rationale,
                    )
                    m_pers, r_pers = agent_cfg["completion_personalization"]
                    t_a = time.perf_counter()
                    r1 = call_openrouter(
                        api_key,
                        m_pers,
                        m1,
                        temperature=LLM_TEMPERATURE,
                        use_json_object=True,
                        pipeline=st.session_state.pipeline,
                        agent_name="completion_personalization",
                        reasoning_effort=r_pers,
                    )
                    lat1 = time.perf_counter() - t_a
                    ts1 = dt.datetime.now(dt.timezone.utc).isoformat(
                        timespec="seconds")
                    st.session_state.starter_welcome_timing[
                        "completion_personalization"] = {
                        "latency_sec": lat1,
                        "ts_utc": ts1,
                        "http_status": r1.status_code,
                    }
                    if r1.status_code != 200:
                        st.session_state.starter_flow_notice = (
                            f"Completion personalization failed (HTTP {r1.status_code}). "
                            "Download **pipeline (JSON)** from the sidebar and inspect "
                            "`completion_personalization`."
                        )
                    else:
                        pers_text = parse_json_single_text(
                            r1.text, "personalized_message"
                        ).strip()
                        st.session_state.starter_personalized = pers_text or None
                        if not pers_text:
                            st.session_state.starter_flow_notice = (
                                "Completion personalization returned HTTP 200 but no usable "
                                "`personalized_message` text was parsed. Check the pipeline export."
                            )
                        m2 = build_session_engagement_messages(
                            output_language=output_language_code,
                            session_json=sess,
                            steps_json=steps_pl,
                        )
                        m_eng, r_eng = agent_cfg["session_engagement"]
                        t_b = time.perf_counter()
                        r2 = call_openrouter(
                            api_key,
                            m_eng,
                            m2,
                            temperature=LLM_TEMPERATURE,
                            use_json_object=True,
                            pipeline=st.session_state.pipeline,
                            agent_name="session_engagement",
                            reasoning_effort=r_eng,
                        )
                        lat2 = time.perf_counter() - t_b
                        ts2 = dt.datetime.now(dt.timezone.utc).isoformat(
                            timespec="seconds")
                        st.session_state.starter_welcome_timing[
                            "session_engagement"] = {
                            "latency_sec": lat2,
                            "ts_utc": ts2,
                            "http_status": r2.status_code,
                        }
                        if r2.status_code != 200:
                            msg2 = (
                                f"Session engagement failed (HTTP {r2.status_code}). "
                                "Inspect `session_engagement` in the pipeline export."
                            )
                            if st.session_state.starter_flow_notice:
                                st.session_state.starter_flow_notice += " — " + msg2
                            else:
                                st.session_state.starter_flow_notice = msg2
                        else:
                            eng_text = parse_json_single_text(
                                r2.text, "engagement_message"
                            ).strip()
                            st.session_state.starter_engagement = eng_text or None
                            if not eng_text:
                                hint = (
                                    "Session engagement returned HTTP 200 but no usable "
                                    "`engagement_message` was parsed (wrong JSON shape or empty). "
                                    "Open the pipeline JSON and read `response.raw_text` for "
                                    "`session_engagement`."
                                )
                                if st.session_state.starter_flow_notice:
                                    st.session_state.starter_flow_notice += (
                                        " — " + hint
                                    )
                                else:
                                    st.session_state.starter_flow_notice = hint
                    st.rerun()

        pending_skill_q = st.text_input(
            "Pending skill question (optional, session agent only)",
            value="",
            help="If set, appended to session rules as in production.",
            key="pending_skill_question",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat + pipeline"):
                st.session_state.messages = []
                st.session_state.pipeline = []
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.upload_session = None
                st.session_state.upload_steps = None
                st.session_state.starter_personalized = None
                st.session_state.starter_engagement = None
                st.session_state.starter_welcome_timing = {}
                st.session_state.starter_flow_notice = ""
                st.rerun()
        with c2:
            st.caption(
                f"conversation_id: `{st.session_state.conversation_id[:8]}…`")

        export_obj = {
            "meta": {
                "agents": {
                    k: {"model": v[0], "reasoning_effort": v[1]}
                    for k, v in agent_cfg.items()
                },
                "conversation_id": st.session_state.conversation_id,
                "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "steps": st.session_state.pipeline,
        }
        st.download_button(
            "Download pipeline (JSON)",
            data=json.dumps(export_obj, ensure_ascii=False, indent=2),
            file_name="openrouter_pipeline.json",
            mime="application/json",
        )

    st.title("Intent + Session agents (OpenRouter)")
    st.caption(
        "Chat uses the same `recent_messages` slice (last 6) for intent and session agents; "
        "each agent’s model and reasoning effort are set in the sidebar.")

    notice0 = (st.session_state.get("starter_flow_notice") or "").strip()
    if notice0:
        st.warning(notice0)

    render_welcome_flow_blocks()

    max_gap, max_latency = chat_timing_maxima(st.session_state.messages)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            cs = m.get("chart_spec")
            if cs:
                render_chart_spec(cs)
            render_chat_message_timing(
                m, max_gap=max_gap, max_latency=max_latency)

    msgs = st.session_state.messages
    user_gaps = [
        float(m["gap_since_prev_sec"])
        for m in msgs
        if m.get("role") == "user"
        and isinstance(m.get("gap_since_prev_sec"), (int, float))
    ]
    asst_lat = [
        float(m["response_time_sec"])
        for m in msgs
        if m.get("role") == "assistant"
        and isinstance(m.get("response_time_sec"), (int, float))
    ]
    if user_gaps or asst_lat:
        import plotly.express as px

        with st.expander("Timing overview (charts)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                if user_gaps:
                    st.caption("User turns: seconds since previous chat message")
                    fig_g = px.bar(
                        x=list(range(1, len(user_gaps) + 1)),
                        y=user_gaps,
                        labels={"x": "User turn #", "y": "Seconds"},
                    )
                    fig_g.update_layout(
                        height=280, margin=dict(t=8, b=8, l=8, r=8))
                    st.plotly_chart(fig_g, use_container_width=True)
            with c2:
                if asst_lat:
                    st.caption("Assistant replies: end-to-end latency")
                    fig_a = px.bar(
                        x=list(range(1, len(asst_lat) + 1)),
                        y=asst_lat,
                        labels={"x": "Reply #", "y": "Seconds"},
                    )
                    fig_a.update_layout(
                        height=280, margin=dict(t=8, b=8, l=8, r=8))
                    st.plotly_chart(fig_a, use_container_width=True)

    if not api_key:
        st.info("Configure `OPENROUTER_API_KEY` in secrets to send messages.")
        return

    user_text = st.chat_input("Message")
    if not user_text:
        return

    prior_messages = list(st.session_state.messages)
    t_send = dt.datetime.now(dt.timezone.utc)
    gap_since_prev_sec: float | None = None
    if prior_messages:
        prev = prior_messages[-1]
        pts = prev.get("ts_utc")
        if isinstance(pts, str) and pts.strip():
            try:
                t_prev = dt.datetime.fromisoformat(
                    pts.replace("Z", "+00:00"))
                if t_prev.tzinfo is None:
                    t_prev = t_prev.replace(tzinfo=dt.timezone.utc)
                gap_since_prev_sec = max(
                    0.0,
                    (
                        t_send - t_prev.astimezone(dt.timezone.utc)
                    ).total_seconds(),
                )
            except ValueError:
                gap_since_prev_sec = None
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
            "ts_utc": t_send.isoformat(timespec="seconds"),
            **(
                {"gap_since_prev_sec": gap_since_prev_sec}
                if gap_since_prev_sec is not None
                else {}
            ),
        }
    )
    prior_hist = messages_to_agent_history(prior_messages)
    recent = prior_hist[-6:] if len(prior_hist) > 6 else prior_hist
    t_reply_start = time.perf_counter()

    parts: list[str] = []
    pending_chart_spec: dict[str, Any] | None = None
    session_payload = st.session_state.get("upload_session")
    steps_payload = st.session_state.get("upload_steps")
    m_vis, r_vis = agent_cfg["visualization_agent"]
    m_sess, r_sess = agent_cfg["session_question"]
    m_intent, r_intent = agent_cfg["intent_classifier"]

    def _run_visualization(source_label: str) -> None:
        nonlocal pending_chart_spec
        vmsgs = build_visualization_messages(
            question=user_text,
            session_json=session_payload,
            steps_json=steps_payload,
            recent_history=recent,
            pending_skill_question=pending_skill_q.strip() or None,
        )
        vres = call_openrouter(
            api_key,
            m_vis,
            vmsgs,
            temperature=LLM_TEMPERATURE,
            use_json_object=True,
            pipeline=st.session_state.pipeline,
            agent_name="visualization_agent",
            reasoning_effort=r_vis,
        )
        if vres.status_code != 200:
            parts.append(
                f"Visualization agent HTTP {vres.status_code}. Check pipeline export.")
        else:
            v_answer, v_chart = parse_visualization_response(vres.text)
            parts.append(v_answer)
            pending_chart_spec = v_chart

    if explicit_intent == "skill_creation":
        parts.append(
            "**Intent routing:** `skill_creation` — demo has no extra LLM for this path; "
            "configure skill flow in your backend."
        )
    elif explicit_intent == "visualization_request":
        _run_visualization("direct")
    elif explicit_intent == "session_question":
        msgs = build_session_messages(
            question=user_text,
            session_json=session_payload,
            steps_json=steps_payload,
            recent_history=recent,
            pending_skill_question=pending_skill_q.strip() or None,
        )
        res = call_openrouter(
            api_key,
            m_sess,
            msgs,
            temperature=LLM_TEMPERATURE,
            use_json_object=True,
            pipeline=st.session_state.pipeline,
            agent_name="session_question",
            reasoning_effort=r_sess,
        )
        if res.status_code != 200:
            parts.append(
                f"Session agent HTTP {res.status_code}. Check pipeline export for details.")
        else:
            parts.append(parse_session_answer(res.text))
    elif explicit_intent == "general_chat":
        msgs = build_intent_classifier_messages(user_text, recent)
        res = call_openrouter(
            api_key,
            m_intent,
            msgs,
            temperature=LLM_TEMPERATURE,
            use_json_object=True,
            pipeline=st.session_state.pipeline,
            agent_name="intent_classifier",
            reasoning_effort=r_intent,
        )
        resolved = parse_intent(res.text) if res.status_code == 200 else None
        if resolved is None:
            resolved = "general_chat"
            src = "fallback"
        else:
            src = "llm"
        parts.append(f"**Classifier ({src}):** `{resolved}`")

        if resolved == "visualization_request":
            _run_visualization("classifier")
        elif resolved == "session_question":
            msgs2 = build_session_messages(
                question=user_text,
                session_json=session_payload,
                steps_json=steps_payload,
                recent_history=recent,
                pending_skill_question=pending_skill_q.strip() or None,
            )
            res2 = call_openrouter(
                api_key,
                m_sess,
                msgs2,
                temperature=LLM_TEMPERATURE,
                use_json_object=True,
                pipeline=st.session_state.pipeline,
                agent_name="session_question",
                reasoning_effort=r_sess,
            )
            if res2.status_code != 200:
                parts.append(f"Session agent HTTP {res2.status_code}.")
            else:
                parts.append(parse_session_answer(res2.text))

    assistant_text = "\n\n".join(parts)
    response_time_sec = time.perf_counter() - t_reply_start
    t_done = dt.datetime.now(dt.timezone.utc)
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text,
        "response_time_sec": response_time_sec,
        "ts_utc": t_done.isoformat(timespec="seconds"),
    }
    if pending_chart_spec is not None:
        assistant_msg["chart_spec"] = pending_chart_spec
    st.session_state.messages.append(assistant_msg)
    st.rerun()


if __name__ == "__main__":
    main()
