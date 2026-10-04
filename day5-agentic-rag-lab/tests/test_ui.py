"""
Browser end-to-end tests for the chat UI (README §8.4), using Playwright + the mock LLM.
"""
import re

import pytest
from playwright.sync_api import expect

from conftest import FIXTURES, PROJECT_ROOT, ask, tid

pytestmark = pytest.mark.ui


def last(page, name):
    return page.locator(tid(name)).last


def test_ui_01_page_loads(chat_page):
    assert "Ask My Documents" in chat_page.title()
    expect(chat_page.locator(tid("health-indicator"))).to_have_attribute("data-status", "ok")
    expect(chat_page.locator(tid("chunk-count"))).to_have_text(re.compile(r"^[1-9]\d*$"))


def test_ui_02_mode_selector(chat_page):
    values = chat_page.locator(f'{tid("mode-select")} option').evaluate_all("opts => opts.map(o => o.value)")
    assert values == ["rag", "agent_scratch", "agent_graph"]


def test_ui_03_send_button_state(chat_page):
    send = chat_page.locator(tid("send-button"))
    expect(send).to_be_disabled()
    chat_page.fill(tid("chat-input"), "Hello")
    expect(send).to_be_enabled()
    chat_page.fill(tid("chat-input"), "   ")
    expect(send).to_be_disabled()


def test_ui_04_rag_answer_with_sources(chat_page):
    ask(chat_page, "How many days of Casual Leave per year?", "rag")
    expect(chat_page.locator(tid("user-message"))).to_have_count(1)
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(1)
    expect(last(chat_page, "sources-list")).to_contain_text("hr_leave_policy.md")
    expect(last(chat_page, "latency")).to_be_visible()


def test_ui_05_off_topic(chat_page):
    ask(chat_page, "What is the cafeteria menu?", "rag")
    expect(last(chat_page, "assistant-message")).to_contain_text("don't know")


def test_ui_06_scratch_agent_uses_calculator(chat_page):
    ask(chat_page, "What is 18% of 4520 plus 75?", "agent_scratch")
    last(chat_page, "trace-toggle").click()
    expect(last(chat_page, "trace-panel")).to_be_visible()
    expect(chat_page.locator(tid("tool-badge"), has_text="calculator")).to_have_count(1)


def test_ui_07_scratch_agent_uses_retriever(chat_page):
    ask(chat_page, "Who approves a 10-day leave request?", "agent_scratch")
    last(chat_page, "trace-toggle").click()
    expect(chat_page.locator(tid("tool-badge"), has_text="search_documents")).to_have_count(1)


def test_ui_08_langgraph_agent(chat_page):
    ask(chat_page, "What is 18% of 4520 plus 75?", "agent_graph")
    last(chat_page, "trace-toggle").click()
    expect(last(chat_page, "trace-panel")).to_be_visible()
    expect(chat_page.locator(tid("tool-badge"), has_text="calculator")).to_have_count(1)


def test_ui_09_trace_toggle_collapses(chat_page):
    ask(chat_page, "What is 18% of 4520 plus 75?", "agent_scratch")
    toggle, panel = last(chat_page, "trace-toggle"), last(chat_page, "trace-panel")
    expect(panel).to_be_hidden()
    toggle.click()
    expect(panel).to_be_visible()
    toggle.click()
    expect(panel).to_be_hidden()


def test_ui_10_multi_turn(chat_page):
    ask(chat_page, "How long is Paternity Leave?", "rag")
    ask(chat_page, "What is the cafeteria menu?", "rag")
    expect(chat_page.locator(tid("user-message"))).to_have_text(["How long is Paternity Leave?", "What is the cafeteria menu?"])
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(2)
    expect(chat_page.locator(tid("assistant-message")).last).to_contain_text("don't know")


def test_ui_11_enter_key_sends(chat_page):
    chat_page.fill(tid("chat-input"), "How long is Paternity Leave?")
    chat_page.press(tid("chat-input"), "Enter")
    expect(chat_page.locator(tid("user-message"))).to_have_count(1)
    expect(chat_page.locator(tid("chat-input"))).to_have_value("")
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(1)


def test_ui_12_thinking_indicator(chat_page):
    held = []
    chat_page.route("**/chat", lambda route: held.append(route))  # hold the request
    chat_page.fill(tid("chat-input"), "How long is Paternity Leave?")
    chat_page.click(tid("send-button"))
    indicator = chat_page.locator(tid("thinking-indicator"))
    expect(indicator).to_be_visible()
    chat_page.wait_for_timeout(2000)  # the response is delayed by 2 s...
    expect(indicator).to_be_visible()  # ...and the indicator stays up the whole time
    expect(chat_page.locator(tid("send-button"))).to_be_disabled()
    held[0].continue_()
    expect(indicator).to_be_hidden()
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(1)


def test_ui_13_backend_error(chat_page):
    chat_page.route("**/chat", lambda route: route.fulfill(status=500, content_type="application/json",
                                                         body='{"error": "Simulated failure"}'))
    ask(chat_page, "How long is Paternity Leave?", "rag")
    expect(chat_page.locator(tid("error-message"))).to_be_visible()
    expect(chat_page.locator(tid("error-message"))).to_contain_text("Simulated failure")
    chat_page.unroute("**/chat")
    ask(chat_page, "How long is Paternity Leave?", "rag")
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(1)


def test_ui_14_upload_flow(chat_page, clean_docs):
    before = int(chat_page.text_content(tid("chunk-count")))
    chat_page.set_input_files(tid("upload-input"), str(FIXTURES / "wfh_policy.md"))
    expect(chat_page.locator(tid("upload-status"))).to_contain_text("Added", timeout=60_000)
    expect(chat_page.locator(tid("chunk-count"))).not_to_have_text(str(before))
    assert int(chat_page.text_content(tid("chunk-count"))) > before
    ask(chat_page, "How many WFH days per week are allowed?", "rag")
    expect(last(chat_page, "sources-list")).to_contain_text("wfh_policy.md")


def test_ui_15_mobile_layout(page):
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto("/")
    page.wait_for_selector(f'{tid("health-indicator")}[data-status="ok"]')
    assert page.evaluate("document.documentElement.scrollWidth") <= 375
    expect(page.locator(tid("chat-input"))).to_be_visible()
    expect(page.locator(tid("send-button"))).to_be_visible()


# ---------------------------------------------------------------------------
# Short-term memory (Step 6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["agent_scratch", "agent_graph"])
def test_ui_16_agent_remembers_follow_up(chat_page, mode):
    ask(chat_page, "How long is Paternity Leave?", mode)
    ask(chat_page, "Can it be split into two parts?", mode)
    last(chat_page, "trace-toggle").click()
    expect(last(chat_page, "trace-panel")).to_contain_text("Paternity")
    expect(last(chat_page, "memory-indicator")).to_be_visible()


def test_ui_17_new_chat_forgets(chat_page):
    ask(chat_page, "How long is Paternity Leave?", "agent_graph")
    chat_page.click(tid("new-chat"))
    expect(chat_page.locator(tid("user-message"))).to_have_count(0)
    expect(chat_page.locator(tid("assistant-message"))).to_have_count(0)
    ask(chat_page, "Can it be split into two parts?", "agent_graph")
    last(chat_page, "trace-toggle").click()
    expect(last(chat_page, "trace-panel")).to_be_visible()
    expect(last(chat_page, "trace-panel")).not_to_contain_text("Paternity Leave? Can it")
    expect(chat_page.locator(tid("memory-indicator"))).to_have_count(0)


def test_ui_18_upload_pdf_twin_is_not_duplicated(chat_page, clean_docs):
    before = chat_page.text_content(tid("chunk-count"))
    chat_page.set_input_files(tid("upload-input"), str(PROJECT_ROOT / "sample_docs" / "hr_leave_policy.pdf"))
    expect(chat_page.locator(tid("upload-status"))).to_contain_text("already indexed", timeout=60_000)
    chat_page.wait_for_timeout(500)
    expect(chat_page.locator(tid("chunk-count"))).to_have_text(before)
