from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ...config import Settings
from ...costs_ledger import chat_llm_context
from ...service import ServiceError
from ..llm_client import call_openrouter_ex, supports_json_format
from .llm_language import SYSTEM_LANGUAGE_DIRECTIVE

SUPPORTED_INTENTS = frozenset({"skill_creation", "general_chat", "session_question"})


@dataclass
class IntentDecision:
    intent: str
    source: str
    confidence: float
    llm_cost: dict[str, Any] | None = None


class SkillIntentAgent:
    """
    Resolves intent. If explicit_intent is 'skill_creation', uses it.
    If explicit_intent is 'general_chat', uses LLM to distinguish between 'general_chat' (small talk) and 'session_question' (questions about the report).
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve_intent(
        self,
        *,
        explicit_intent: str | None,
        message: str | None,
        history: list[dict[str, str]] | None,
        context: dict[str, object] | None,
        conversation_id: str | None = None,
    ) -> IntentDecision:
        _ = (context,)
        raw = (explicit_intent or "").strip().lower()
        if not raw:
            raise ServiceError("missing_intent", "intent is required for kind='intent' requests", 422)
        if raw not in SUPPORTED_INTENTS:
            raise ServiceError(
                "invalid_intent",
                f"intent must be one of {', '.join(SUPPORTED_INTENTS)}",
                422,
            )
        
        # Fast path for skill_creation or session_question (if client already knows)
        if raw in {"skill_creation", "session_question"}:
            return IntentDecision(intent=raw, source="api", confidence=1.0)
        
        # If 'general_chat', let's use LLM to see if it's actually a 'session_question'
        msg = (message or "").strip()
        if not msg or not self.settings.openrouter_api_key.strip():
            return IntentDecision(intent=raw, source="api", confidence=1.0)
        
        return self._classify_chat_intent(msg, history or [], conversation_id=conversation_id)

    def _classify_chat_intent(self, message: str, history: list[dict[str, str]], conversation_id: str | None = None) -> IntentDecision:
        prompt = {
            "task": "Classify the user's message intent.",
            "rules": [
                "If the user asks about tasks, actions, goals, or metrics within the current session or report, output 'session_question'.",
                "If the user is just saying hello, asking general questions, or small-talking, output 'general_chat'.",
            ],
            "recent_messages": history[-6:],
            "user_message": message,
            "output": {"intent": "general_chat | session_question"}
        }

        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_LANGUAGE_DIRECTIVE} Return JSON only. You are an intent classifier.",
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
                temperature=0,
                extra_body=extra_body,
                cost_context=chat_llm_context(conversation_id, "intent_classifier"),
            )
            parsed = json.loads(llm.text)
            intent = str(parsed.get("intent") or "").strip().lower()
            if intent in {"session_question", "general_chat"}:
                return IntentDecision(intent=intent, source="llm", confidence=1.0, llm_cost=llm.cost)
        except Exception:
            pass

        return IntentDecision(intent="general_chat", source="fallback", confidence=1.0)
