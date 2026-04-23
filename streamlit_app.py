"""
Autonomous Streamlit app: Intent classifier + Session question agents via OpenRouter.
Uploaded JSON is passed to the model as provided (no field validation or stripping).
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
# session_question_agent helpers
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


SUPPORTED_INTENTS = frozenset({"skill_creation", "general_chat", "session_question"})

# Defaults for OpenRouter completion calls (session + intent agents)
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-pro"
DEFAULT_COMPLETION_PERSONALIZATION_MODEL = "openai/gpt-4o-mini"
DEFAULT_SESSION_ENGAGEMENT_MODEL = "google/gemini-2.5-pro"
LLM_TEMPERATURE = 0.1
COMPLETION_AGENT_REASONING_EFFORT = "high"

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
        "reasoning": {"effort": COMPLETION_AGENT_REASONING_EFFORT},
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
        minimal = {k: v for k, v in payload.items() if k not in ("response_format", "reasoning")}
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
        if intent in {"session_question", "general_chat"}:
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
            status = str(item.get("goal_completion_status") or "").strip().lower()
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
            "Rewrite the session completion rationale into one short, warm, personalized line for the user."
        ),
        "inputs": {
            "user_name": user_name,
            "output_language": output_language,
            "completion_rationale": completion_rationale,
        },
        "rules": [
            "Greet the user by name naturally (spoken style, not formal ledgers).",
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
            DEFAULT_OPENROUTER_MODEL,
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
        ]
        choice = st.selectbox("Model", ["Custom"] + presets, index=1, key="model_choice")
        if choice == "Custom":
            model = st.text_input(
                "Custom model id",
                value=DEFAULT_OPENROUTER_MODEL,
                key="custom_model",
            )
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
                st.session_state.upload_session = json.loads(f_sess.getvalue().decode("utf-8"))
            except json.JSONDecodeError as e:
                st.error(f"Session file is not valid JSON: {e}")

        if f_steps is not None:
            try:
                st.session_state.upload_steps = json.loads(f_steps.getvalue().decode("utf-8"))
            except json.JSONDecodeError as e:
                st.error(f"Steps file is not valid JSON: {e}")

        st.subheader("Welcome agents (pre-chat)")
        starter_user_name = st.text_input("User name", value="Peter", key="starter_user_name")
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
                rationale = extract_first_complete_completion_rationale(sess) if sess is not None else None
                if sess is None:
                    st.error("Upload a Session (JSON) file first.")
                elif not rationale:
                    st.error(
                        "No completion_rationale found for a goal with goal_completion_status "
                        "'complete' in the session JSON."
                    )
                else:
                    st.session_state.starter_personalized = None
                    st.session_state.starter_engagement = None
                    m1 = build_completion_personalization_messages(
                        user_name=starter_user_name.strip() or "Peter",
                        output_language=output_language_code,
                        completion_rationale=rationale,
                    )
                    r1 = call_openrouter(
                        api_key,
                        DEFAULT_COMPLETION_PERSONALIZATION_MODEL,
                        m1,
                        temperature=LLM_TEMPERATURE,
                        use_json_object=True,
                        pipeline=st.session_state.pipeline,
                        agent_name="completion_personalization",
                    )
                    if r1.status_code != 200:
                        st.error(f"Completion personalization agent HTTP {r1.status_code}. See pipeline export.")
                    else:
                        st.session_state.starter_personalized = parse_json_single_text(
                            r1.text, "personalized_message"
                        )
                        m2 = build_session_engagement_messages(
                            output_language=output_language_code,
                            session_json=sess,
                            steps_json=steps_pl,
                        )
                        r2 = call_openrouter(
                            api_key,
                            DEFAULT_SESSION_ENGAGEMENT_MODEL,
                            m2,
                            temperature=LLM_TEMPERATURE,
                            use_json_object=True,
                            pipeline=st.session_state.pipeline,
                            agent_name="session_engagement",
                        )
                        if r2.status_code != 200:
                            st.error(f"Session engagement agent HTTP {r2.status_code}. See pipeline export.")
                        else:
                            st.session_state.starter_engagement = parse_json_single_text(
                                r2.text, "engagement_message"
                            )
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

    sp0 = st.session_state.get("starter_personalized")
    se0 = st.session_state.get("starter_engagement")
    if sp0 or se0:
        st.subheader("Welcome flow (from Start)")
        if sp0:
            st.markdown(sp0)
        if se0:
            st.markdown(se0)
        st.divider()

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
    session_payload = st.session_state.get("upload_session")
    steps_payload = st.session_state.get("upload_steps")
    model_used = st.session_state.get("model", model)

    if explicit_intent == "skill_creation":
        parts.append(
            "**Intent routing:** `skill_creation` — demo has no extra LLM for this path; "
            "configure skill flow in your backend."
        )
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
            model_used,
            msgs,
            temperature=LLM_TEMPERATURE,
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
            temperature=LLM_TEMPERATURE,
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
                session_json=session_payload,
                steps_json=steps_payload,
                recent_history=recent,
                pending_skill_question=pending_skill_q.strip() or None,
            )
            res2 = call_openrouter(
                api_key,
                model_used,
                msgs2,
                temperature=LLM_TEMPERATURE,
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
