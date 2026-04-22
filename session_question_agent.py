from __future__ import annotations

import json
import logging
from typing import Any

from ...config import Settings
from ...costs_ledger import chat_llm_context
from ..llm_client import call_openrouter_ex, supports_json_format
from .llm_language import SYSTEM_LANGUAGE_DIRECTIVE

logger = logging.getLogger(__name__)

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


# Per-field truncation budgets (characters).
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


class SessionQuestionAgent:
    """Answers user questions strictly based on session/report data."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def answer_question(
        self,
        *,
        question: str,
        session_data: Any,
        raw_steps: list[dict[str, Any]],
        recent_history: list[dict[str, str]],
        pending_skill_question: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        if not self.settings.openrouter_api_key.strip():
            return "No API key configured.", None

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

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_LANGUAGE_DIRECTIVE} Return JSON only. You are an analytical session assistant.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        extra_body = (
            {"response_format": {"type": "json_object"}}
            if supports_json_format(self.settings.openrouter_model)
            else None
        )

        try:
            llm = call_openrouter_ex(
                self.settings,
                messages,
                model=self.settings.openrouter_model,
                temperature=0.2,
                extra_body=extra_body,
                cost_context=chat_llm_context(conversation_id, "session_question"),
            )
        except Exception as e:
            logger.exception(
                "SessionQuestionAgent LLM call failed: conversation_id=%s steps=%d err=%s",
                conversation_id,
                len(raw_steps or []),
                e,
            )
            return (
                "I could not retrieve an answer about this session right now. Please try again in a moment.",
                None,
            )

        raw_text = llm.text or ""
        answer = ""
        try:
            parsed = json.loads(raw_text)
            answer = _extract_answer_from_parsed(parsed)
        except json.JSONDecodeError:
            logger.warning(
                "SessionQuestionAgent non-JSON LLM response: conversation_id=%s steps=%d preview=%s",
                conversation_id,
                len(raw_steps or []),
                _preview_llm_text(raw_text),
            )
            answer = raw_text.strip()
        except Exception as e:
            logger.exception(
                "SessionQuestionAgent parse error: conversation_id=%s shape=%s err=%s preview=%s",
                conversation_id,
                type(parsed).__name__ if "parsed" in locals() else "unknown",
                e,
                _preview_llm_text(raw_text),
            )
            answer = raw_text.strip()

        if not answer:
            logger.warning(
                "SessionQuestionAgent empty answer after parsing: conversation_id=%s preview=%s",
                conversation_id,
                _preview_llm_text(raw_text),
            )
            answer = "I could not generate an answer from the session data."

        return answer, llm.cost