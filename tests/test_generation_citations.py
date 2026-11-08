import asyncio
import json

import pytest

from core.config import settings
from service import generation_service
from service.citation_service import select_cited_chunks

CONTEXT = [
    {"doc_title": "Бюджет", "text": "Бюджет — 100 рублей."},
    {"doc_title": "План проекта", "text": "Пилот: 08–10.12.2026. Запуск: 14.12.2026."},
]


def generate(monkeypatch, replies, *, faithfulness=False):
    calls = []
    monkeypatch.setattr(settings, "FAITHFULNESS_CHECK_ENABLED", faithfulness)

    async def llm(prompt, system_prompt=generation_service.SYSTEM_INSTRUCTIONS):
        calls.append((prompt, system_prompt))
        return replies[len(calls) - 1]

    monkeypatch.setattr(generation_service, "_ask_llm", llm)
    answer = asyncio.run(generation_service.generate_answer("Какой план проекта?", CONTEXT))
    return answer, calls


def test_uncited_plan_is_repaired_using_only_its_evidence(monkeypatch):
    draft = "Пилот: 08–10.12.2026. Запуск: 14.12.2026."
    repaired = draft + " [S2]"
    answer, calls = generate(monkeypatch, [draft, json.dumps({"kind": "answer", "answer": repaired})])
    assert answer == repaired
    assert len(calls) == 2
    assert draft in calls[1][0]
    assert '"source": "S2"' in calls[1][0]
    assert "План проекта" in calls[1][0]
    final, sources = select_cited_chunks(answer, CONTEXT)
    assert final == draft + " [S1]"
    assert sources == [CONTEXT[1]]


@pytest.mark.parametrize("draft", ["Ответ без ссылок", "Ответ [S99]", "Ответ [S1] [S99]"])
def test_missing_or_invalid_citations_trigger_one_repair(monkeypatch, draft):
    answer, calls = generate(
        monkeypatch, [draft, '{"kind":"answer","answer":"Запуск: 14.12.2026 [S2]"}']
    )
    assert answer == "Запуск: 14.12.2026 [S2]"
    assert len(calls) == 2


@pytest.mark.parametrize("answer", ["Запуск: 14.12.2026 [S2]", "У меня недостаточно информации"])
def test_cited_answer_or_abstention_does_not_trigger_extra_call(monkeypatch, answer):
    result, calls = generate(monkeypatch, [answer])
    assert result == answer
    assert len(calls) == 1


@pytest.mark.parametrize("repair", [
    "still no citations",
    '{"kind":"answer","answer":"Запуск завтра"}',
    '{"kind":"answer","answer":"Запуск завтра [S99]"}',
    '{"kind":"insufficient","answer":"Неподтверждённое утверждение"}',
    '{"kind":"clarification","answer":"Запуск завтра."}',
    '{"kind":"other","answer":"Запуск завтра [S2]"}',
])
def test_failed_repair_does_not_leak_uncited_factual_answer(monkeypatch, repair):
    answer, calls = generate(monkeypatch, ["Запуск завтра", repair])
    assert answer == "У меня недостаточно информации"
    assert len(calls) == 2


def test_clarifying_question_can_remain_without_sources(monkeypatch):
    question = "Вас интересует пилот или общий запуск?"
    answer, calls = generate(
        monkeypatch, [question, json.dumps({"kind": "clarification", "answer": question})]
    )
    assert answer == question
    assert len(calls) == 2


def test_markdown_wrapped_repair_json_is_accepted(monkeypatch):
    repaired = "Запуск: 14.12.2026 [S2]"
    response = "```json\n" + json.dumps({"kind": "answer", "answer": repaired}) + "\n```"
    answer, calls = generate(monkeypatch, ["Запуск: 14.12.2026", response])
    assert answer == repaired
    assert len(calls) == 2


def test_faithfulness_checks_repaired_answer(monkeypatch):
    repaired = "Запуск: 14.12.2026 [S2]"
    answer, calls = generate(
        monkeypatch, ["Запуск: 14.12.2026", json.dumps({"kind": "answer", "answer": repaired}), "UNSUPPORTED"],
        faithfulness=True,
    )
    assert answer == "У меня недостаточно информации"
    assert repaired in calls[2][0]


def test_empty_context_does_not_call_model(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("No LLM call is needed without context")

    monkeypatch.setattr(generation_service, "_ask_llm", unexpected)
    assert asyncio.run(generation_service.generate_answer("question", [])) == "У меня недостаточно информации"
