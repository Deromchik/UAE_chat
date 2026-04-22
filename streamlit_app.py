"""
Autonomous Streamlit app: Intent classifier + Session question agents via OpenRouter.
All prompts and compaction logic are inlined (copied from UAE_chat agent modules).
"""
from __future__ import annotations

import datetime as dt
import json
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
# session_question_agent helpers (verbatim)
# ---------------------------------------------------------------------------
_ANSWER_KEYS = ("assistant_answer", "answer", "response", "reply", "message", "text")


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


TEXT_SHORT = 200
TEXT_MED = 500
EMAIL_SUMMARY_MAX = 600
SPREADSHEET_GRID_PREVIEW_ROWS = 3
SPREADSHEET_GRID_PREVIEW_COLS = 6
SPREADSHEET_GRID_CELL_MAX = 80
SPREADSHEET_CHANGES_MAX = 20


def _clip(value: Any, limit: int) -> str | None:
    """Coerce to string and clip, drop empty / None values."""
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return None
    value = value.strip()
    if not value:
        return None
    if len(value) > limit:
        return value[:limit].rstrip() + "…"
    return value


def _first_nonempty(*values: Any) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _compact_mail_content(mail: dict[str, Any]) -> dict[str, Any]:
    """Extract useful email fields, drop heavy/DOM-ish ones."""
    to_list = [_clip(x, TEXT_SHORT) for x in _as_list(mail.get("to"))]
    cc_list = [_clip(x, TEXT_SHORT) for x in _as_list(mail.get("cc"))]
    bcc_list = [_clip(x, TEXT_SHORT) for x in _as_list(mail.get("bcc"))]

    out: dict[str, Any] = {
        "status": _clip(mail.get("status"), TEXT_SHORT),
        "source": _clip(mail.get("source"), TEXT_SHORT),
        "message_id": _clip(mail.get("messageId") or mail.get("message_id"), TEXT_SHORT),
        "thread_id": _clip(mail.get("threadId") or mail.get("thread_id"), TEXT_SHORT),
        "subject": _clip(mail.get("subject"), TEXT_SHORT),
        "from": _clip(mail.get("from_email") or mail.get("from"), TEXT_SHORT),
        "to": [x for x in to_list if x][:10] or None,
        "cc": [x for x in cc_list if x][:10] or None,
        "bcc": [x for x in bcc_list if x][:10] or None,
        "sent_at": _clip(mail.get("sent_at"), TEXT_SHORT),
        "summary": _clip(mail.get("content_summary"), EMAIL_SUMMARY_MAX),
    }
    return {k: v for k, v in out.items() if v is not None}


def _compact_spreadsheet(sp: dict[str, Any]) -> dict[str, Any]:
    """Extract spreadsheet step payload without exposing the full grid."""
    data = sp.get("data") if isinstance(sp.get("data"), dict) else {}
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    sheet = data.get("sheet") if isinstance(data.get("sheet"), dict) else {}

    grid_preview: list[list[str]] | None = None
    grid = data.get("grid")
    if isinstance(grid, list) and grid:
        preview: list[list[str]] = []
        for row in grid[:SPREADSHEET_GRID_PREVIEW_ROWS]:
            if not isinstance(row, list):
                continue
            row_cells: list[str] = []
            for cell in row[:SPREADSHEET_GRID_PREVIEW_COLS]:
                clipped = _clip(cell, SPREADSHEET_GRID_CELL_MAX) or ""
                row_cells.append(clipped)
            preview.append(row_cells)
        if preview:
            grid_preview = preview

    def _compact_change(ch: Any) -> dict[str, Any] | None:
        if not isinstance(ch, dict):
            return None
        coords = ch.get("coordinates") if isinstance(ch.get("coordinates"), dict) else {}
        entry = {
            "cell": _clip(
                coords.get("readable")
                or ch.get("readable")
                or ch.get("range"),
                TEXT_SHORT,
            ),
            "row": coords.get("row_index") if isinstance(coords.get("row_index"), int) else ch.get("row_index"),
            "col": coords.get("col_index") if isinstance(coords.get("col_index"), int) else ch.get("col_index"),
            "new_value": _clip(ch.get("new_value"), TEXT_SHORT),
            "old_value": _clip(ch.get("old_value"), TEXT_SHORT),
        }
        entry = {k: v for k, v in entry.items() if v is not None}
        return entry or None

    changes_src = data.get("changes") or data.get("changes_from_previous") or []
    changes_compact: list[dict[str, Any]] = []
    for ch in _as_list(changes_src)[:SPREADSHEET_CHANGES_MAX]:
        compact = _compact_change(ch)
        if compact:
            changes_compact.append(compact)

    out: dict[str, Any] = {
        "provider": _clip(sp.get("provider") or data.get("provider"), TEXT_SHORT),
        "sheet_name": _clip(sp.get("sheetName") or sheet.get("name"), TEXT_SHORT),
        "sheet_id": _clip(sp.get("sheetId") or sheet.get("id"), TEXT_SHORT),
        "prev_sheet_name": _clip(sp.get("prevSheetName"), TEXT_SHORT),
        "prev_sheet_id": _clip(sp.get("prevSheetId"), TEXT_SHORT),
        "doc_url": _clip(sp.get("docUrl") or sp.get("requestUrl"), TEXT_MED),
        "rows": meta.get("rows") if isinstance(meta.get("rows"), int) else None,
        "cols": meta.get("cols") if isinstance(meta.get("cols"), int) else None,
        "changes_count": meta.get("changes_count") if isinstance(meta.get("changes_count"), int) else (len(changes_compact) or None),
        "source_capture_kind": _clip(meta.get("source_capture_kind"), TEXT_SHORT),
        "grid_preview": grid_preview,
        "changes": changes_compact or None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _compact_element(el: dict[str, Any]) -> dict[str, Any]:
    """Extract visible/semantic element info, drop DOM paths and heavy attrs."""
    attrs = el.get("attrs") if isinstance(el.get("attrs"), dict) else {}
    semantic = el.get("semantic") if isinstance(el.get("semantic"), dict) else {}
    locator = el.get("locator") if isinstance(el.get("locator"), dict) else {}
    by_role = locator.get("byRole") if isinstance(locator.get("byRole"), dict) else {}

    aria_name = _first_nonempty(
        attrs.get("aria-label"),
        semantic.get("ariaName"),
        by_role.get("name"),
    )
    out = {
        "text": _clip(el.get("text"), TEXT_MED),
        "aria_name": _clip(aria_name, TEXT_SHORT),
        "title": _clip(attrs.get("title"), TEXT_SHORT),
        "type": _clip(attrs.get("type"), TEXT_SHORT),
    }
    return {k: v for k, v in out.items() if v is not None}


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    """Produce a compact, type-aware projection of a single session step."""
    stype = str(step.get("type") or "")
    base: dict[str, Any] = {
        "id": step.get("stepId") or step.get("step_id"),
        "type": stype or None,
        "time": _clip(step.get("local") or step.get("iso") or step.get("time"), TEXT_SHORT),
        "url": _clip(step.get("url"), TEXT_MED),
        "title": _clip(step.get("title"), TEXT_SHORT),
    }

    tab = step.get("tab") if isinstance(step.get("tab"), dict) else None
    if tab:
        tab_title = _clip(tab.get("title"), TEXT_SHORT)
        if tab_title and tab_title != base.get("title"):
            base["tab_title"] = tab_title

    element = step.get("element") if isinstance(step.get("element"), dict) else None
    if element and stype in {
        "click",
        "button",
        "submit",
        "form_submit",
        "change",
        "email",
        "tel",
        "text",
    }:
        compact_el = _compact_element(element)
        if compact_el:
            base["element"] = compact_el

    input_obj = step.get("input") if isinstance(step.get("input"), dict) else None
    if input_obj:
        value = _clip(input_obj.get("value"), TEXT_MED)
        if value:
            base["input_value"] = value

    mail = step.get("mailContent") if isinstance(step.get("mailContent"), dict) else None
    if mail:
        compact_mail = _compact_mail_content(mail)
        if compact_mail:
            base["mail"] = compact_mail

    sp = step.get("spreadsheet") if isinstance(step.get("spreadsheet"), dict) else None
    if sp:
        compact_sp = _compact_spreadsheet(sp)
        if compact_sp:
            base["spreadsheet"] = compact_sp

    for fallback_key in ("text", "value", "label", "content"):
        fv = step.get(fallback_key)
        if isinstance(fv, str) and fv.strip() and fallback_key not in base:
            base[fallback_key] = _clip(fv, TEXT_MED)

    return {k: v for k, v in base.items() if v not in (None, "", [], {})}


def _slim_goals(goals: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in _as_list(goals):
        if not isinstance(g, dict):
            continue
        sub_actions = []
        for sub in _as_list(g.get("sub_actions")):
            if not isinstance(sub, dict):
                continue
            sub_actions.append({
                "name": _clip(sub.get("sub_action_name") or sub.get("name"), TEXT_MED),
                "status": _clip(sub.get("status"), TEXT_SHORT),
            })
        out.append({
            "goal_id": g.get("goal_id"),
            "goal_name": _clip(g.get("goal_name"), TEXT_MED),
            "status": _clip(g.get("status"), TEXT_SHORT),
            "sub_actions": [s for s in sub_actions if s.get("name")],
        })
    return out


SUPPORTED_INTENTS = frozenset({"skill_creation", "general_chat", "session_question"})


def build_session_messages(
    *,
    question: str,
    session_data: Any,
    raw_steps: list[dict[str, Any]],
    recent_history: list[dict[str, str]],
    pending_skill_question: str | None,
) -> list[dict[str, str]]:
    events_log = [_compact_step(s) for s in (raw_steps or []) if isinstance(s, dict)]
    events_log = [c for c in events_log if c]

    goals = session_data.get("goals") if isinstance(session_data, dict) else session_data if isinstance(session_data, list) else None

    session_context = {
        "status": session_data.get("status") if isinstance(session_data, dict) else None,
        "goals": _slim_goals(goals),
        "metrics": session_data.get("metrics") if isinstance(session_data, dict) else None,
        "total_steps_recorded": len(raw_steps or []),
        "events_log": events_log,
    }

    rules = [
        "Only answer using information present in session_context.",
        "If the answer is not present, clearly state that the information is unavailable.",
        "The 'events_log' contains a chronological list of user actions; each event may include URL, page title, clicked element text/role, and typed input values.",
        "Events with 'mail' describe email content captured during the session: subject, from, to/cc/bcc, dates, summary.",
        "Events with 'spreadsheet' describe Google Sheets or Microsoft Excel activity: provider, sheet name, previous sheet, grid dimensions, a small grid preview, and cell-level changes (cell, new_value, old_value).",
        "The 'goals' list is the high-level structured analysis of what the user accomplished.",
        "Cite concrete values (subjects, recipients, cell changes, URLs, queries) when answering factual questions.",
        "Keep answers concise, factual, professional.",
        "CRITICAL LANGUAGE RULE: Always write the value of 'assistant_answer' in the same natural language as 'user_question'. If the user wrote in Russian, answer in Russian; if in English, answer in English; etc. Never switch language just because session_context keys are in English.",
        "Output format: a single JSON object {\"assistant_answer\": \"...\"}. Do NOT return a JSON array, do NOT wrap the object in a list.",
    ]

    if pending_skill_question:
        rules.append(
            f"CRITICAL: After answering the user's question, you MUST seamlessly transition and ask them the following pending question to resume the skill creation flow: '{pending_skill_question}'"
        )

    prompt = {
        "task": (
            "You are a data-driven assistant analyzing a user's web automation session. "
            "Answer the user's question using ONLY the data available in 'session_context'."
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
        "task": "Classify the user's message intent.",
        "rules": [
            "If the user asks about tasks, actions, goals, or metrics within the current session or report, output 'session_question'.",
            "If the user is just saying hello, asking general questions, or small-talking, output 'general_chat'.",
        ],
        "recent_messages": history[-6:],
        "user_message": message,
        "output": {"intent": "general_chat | session_question"},
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
    if use_json_object and supports_json_format(model):
        payload["response_format"] = {"type": "json_object"}

    def _post(body: dict[str, Any]) -> Any:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=120)
        resp_json: dict[str, Any] | None = None
        try:
            resp_json = r.json()
        except Exception:
            resp_json = None
        raw_text = ""
        if resp_json and isinstance(resp_json, dict):
            choices = resp_json.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
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
    if r.status_code == 400 and "response_format" in payload:
        payload_retry = {k: v for k, v in payload.items() if k != "response_format"}
        r, resp_json, raw_text = _post(payload_retry)

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
        if intent in {"session_question", "general_chat"}:
            return intent
    except Exception:
        pass
    return None


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


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []
    if "session_data" not in st.session_state:
        st.session_state.session_data = {}
    if "raw_steps" not in st.session_state:
        st.session_state.raw_steps = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())


def main() -> None:
    st.set_page_config(page_title="UAE Chat Agents", page_icon="💬", layout="wide")
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

        presets = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
        ]
        choice = st.selectbox("Model", ["Custom"] + presets, index=1, key="model_choice")
        if choice == "Custom":
            model = st.text_input("Custom model id", value="openai/gpt-4o-mini", key="custom_model")
        else:
            model = choice
        st.session_state.model = model

        explicit_intent = st.radio(
            "Explicit intent (routing)",
            sorted(SUPPORTED_INTENTS),
            index=sorted(SUPPORTED_INTENTS).index("general_chat"),
            help="Mirrors API explicit_intent: general_chat runs LLM classifier; session_question skips to session agent.",
        )

        st.subheader("Session data (JSON uploads)")
        f_sess = st.file_uploader(
            "session_data.json — object with status, goals, metrics",
            type=["json"],
            key="session_json",
        )
        f_steps = st.file_uploader(
            "raw_steps.json — array of session step objects",
            type=["json"],
            key="steps_json",
        )

        if f_sess is not None:
            try:
                st.session_state.session_data = json.loads(f_sess.getvalue().decode("utf-8"))
                if not isinstance(st.session_state.session_data, dict):
                    st.error("session_data.json must be a JSON object.")
                    st.session_state.session_data = {}
            except Exception as e:
                st.error(f"Invalid session JSON: {e}")
                st.session_state.session_data = {}

        if f_steps is not None:
            try:
                parsed = json.loads(f_steps.getvalue().decode("utf-8"))
                if not isinstance(parsed, list):
                    st.error("raw_steps.json must be a JSON array.")
                    st.session_state.raw_steps = []
                else:
                    st.session_state.raw_steps = parsed
            except Exception as e:
                st.error(f"Invalid steps JSON: {e}")
                st.session_state.raw_steps = []

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
                st.rerun()
        with c2:
            st.caption(f"conversation_id: `{st.session_state.conversation_id[:8]}…`")

        export_obj = {
            "meta": {
                "model": st.session_state.get("model", model),
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
    st.caption("Chat uses the same `recent_messages` slice (last 6) for intent and session agents.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if not api_key:
        st.info("Configure `OPENROUTER_API_KEY` in secrets to send messages.")
        return

    user_text = st.chat_input("Message")
    if not user_text:
        return

    prior_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_text})
    prior_hist = messages_to_agent_history(prior_messages)
    recent = prior_hist[-6:] if len(prior_hist) > 6 else prior_hist

    parts: list[str] = []
    sess_data = st.session_state.session_data
    steps = st.session_state.raw_steps
    model_used = st.session_state.get("model", model)

    if explicit_intent == "skill_creation":
        parts.append(
            "**Intent routing:** `skill_creation` — demo has no extra LLM for this path; "
            "configure skill flow in your backend."
        )
    elif explicit_intent == "session_question":
        msgs = build_session_messages(
            question=user_text,
            session_data=sess_data,
            raw_steps=steps,
            recent_history=recent,
            pending_skill_question=pending_skill_q.strip() or None,
        )
        res = call_openrouter(
            api_key,
            model_used,
            msgs,
            temperature=0.2,
            use_json_object=True,
            pipeline=st.session_state.pipeline,
            agent_name="session_question",
        )
        if res.status_code != 200:
            parts.append(f"Session agent HTTP {res.status_code}. Check pipeline export for details.")
        else:
            parts.append(parse_session_answer(res.text))
    elif explicit_intent == "general_chat":
        msgs = build_intent_classifier_messages(user_text, recent)
        res = call_openrouter(
            api_key,
            model_used,
            msgs,
            temperature=0.0,
            use_json_object=True,
            pipeline=st.session_state.pipeline,
            agent_name="intent_classifier",
        )
        resolved = parse_intent(res.text) if res.status_code == 200 else None
        if resolved is None:
            resolved = "general_chat"
            src = "fallback"
        else:
            src = "llm"
        parts.append(f"**Classifier ({src}):** `{resolved}`")

        if resolved == "session_question":
            msgs2 = build_session_messages(
                question=user_text,
                session_data=sess_data,
                raw_steps=steps,
                recent_history=recent,
                pending_skill_question=pending_skill_q.strip() or None,
            )
            res2 = call_openrouter(
                api_key,
                model_used,
                msgs2,
                temperature=0.2,
                use_json_object=True,
                pipeline=st.session_state.pipeline,
                agent_name="session_question",
            )
            if res2.status_code != 200:
                parts.append(f"Session agent HTTP {res2.status_code}.")
            else:
                parts.append(parse_session_answer(res2.text))

    assistant_text = "\n\n".join(parts)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()


if __name__ == "__main__":
    main()
