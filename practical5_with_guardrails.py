"""Practical 5 RAG agent demonstration with input, tool, and output guardrails."""

import re
from typing import Any

import practical5_without_guardrails as base_agent
from test_prompts import TEST_PROMPTS


# Input guardrail: detect common attempts to override the agent instructions.
def detect_prompt_injection(message: str) -> str | None:
    patterns = (
        r"\bignore\b.{0,40}\b(previous|prior|all)\b.{0,40}\binstructions?\b",
        r"\breveal\b.{0,40}\b(system prompt|secrets?|api key)\b",
        r"\bpretend\b.{0,30}\byou are\b",
    )
    if any(re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
        return "prompt injection detected"
    return None


# Input guardrail: restrict questions to the Nimbus handbook/RAG domain.
def check_scope(message: str) -> str | None:
    scope_terms = (
        "nimbus", "refund", "pricing", "plan", "support", "security", "compliance",
        "onboarding", "product", "deploy", "monitor", "vault", "customer", "handbook",
        "data", "company", "policy",
    )
    if not any(term in message.lower() for term in scope_terms):
        return "request is outside the Nimbus handbook scope"
    return None


# Input guardrail: prevent excessively large prompts from consuming the context window.
def check_message_length(message: str, maximum: int = 1000) -> str | None:
    if len(message) > maximum:
        return f"message exceeds the {maximum}-character limit"
    return None


# Input guardrail: reject abusive language before it reaches the model.
def check_profanity(message: str) -> str | None:
    profanity_terms = {"stupid", "useless", "damn", "idiot", "fuck", "shit"}
    words = set(re.findall(r"[a-z']+", message.lower()))
    if words.intersection(profanity_terms):
        return "abusive or profane language is not allowed"
    return None


def run_input_guardrails(message: str) -> str | None:
    for check in (
        detect_prompt_injection,
        check_message_length,
        check_profanity,
        check_scope,
    ):
        reason = check(message)
        if reason:
            return f"BLOCKED by input guardrail: {reason}."
    return None


# Tool/action guardrail: this agent is read-only and cannot delete data or change controls.
def check_tool_action(message: str, response: str) -> str | None:
    risky_actions = (
        "delete", "disable", "destroy", "drop", "remove all", "change security",
        "turn off encryption", "erase",
    )
    if any(action in message.lower() for action in risky_actions):
        return "read-only policy forbids destructive or security-control actions"
    return None


# Output guardrail: redact common email and phone-number formats.
def redact_pii(response: str) -> str:
    response = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[REDACTED EMAIL]",
        response,
        flags=re.IGNORECASE,
    )
    return re.sub(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", "[REDACTED PHONE]", response)


# Output guardrail: prevent disclosure of prompts, credentials, or secret-like values.
def block_system_prompt_or_secrets(response: str) -> str | None:
    secret_terms = ("system prompt", "api key", "secret key", "password", "token")
    if any(term in response.lower() for term in secret_terms):
        return "output contains a system-prompt or secret disclosure"
    return None


# Output guardrail: reject toxic or unsafe generated content.
def block_toxic_or_unsafe_content(response: str) -> str | None:
    unsafe_terms = ("how to harm", "make a weapon", "hate speech", "kill someone")
    if any(term in response.lower() for term in unsafe_terms):
        return "output contains toxic or unsafe content"
    return None


# Output guardrail: require a non-empty, bounded text response.
def validate_output_format(response: str) -> str | None:
    if not response.strip():
        return "output is empty"
    if len(response) > 12000:
        return "output exceeds the response-size limit"
    return None


def run_output_guardrails(response: str) -> str:
    response = redact_pii(response)
    for check in (block_system_prompt_or_secrets, block_toxic_or_unsafe_content, validate_output_format):
        reason = check(response)
        if reason:
            return f"BLOCKED by output guardrail: {reason}."
    return response


def guarded_answer(question: str, retriever: Any, llm: Any) -> str:
    input_failure = run_input_guardrails(question)
    if input_failure:
        return input_failure

    # The LLM call remains after all input checks, matching the required order.
    raw_response = base_agent.answer_question(question, retriever, llm)

    tool_failure = check_tool_action(question, raw_response)
    if tool_failure:
        return f"BLOCKED by tool/action guardrail: {tool_failure}."

    return run_output_guardrails(raw_response)


def main() -> None:
    retriever, llm = base_agent.load_agent_components(base_agent.config.DEFAULT_K)
    print("PRACTICAL 5 - WITH GUARDRAILS")
    for test_name, user_input in TEST_PROMPTS:
        response = guarded_answer(user_input, retriever, llm)
        print(f"\nTest: {test_name}\nUser input: {user_input}\nAgent response: {response}\n")


if __name__ == "__main__":
    main()
