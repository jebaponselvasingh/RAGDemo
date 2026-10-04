"""
Chunking & Vector DB Explorer tests (README §8.4A): API checks (EX-*) and browser checks (UI-EX-*).
The Explorer never calls the LLM, so everything runs with the mock provider.
"""
import base64
import io

import pytest
import requests
from playwright.sync_api import expect

from conftest import FIXTURES, PROJECT_ROOT, tid

pytestmark = pytest.mark.ui

SAMPLE_PDF = PROJECT_ROOT / "sample_docs" / "hr_leave_policy.pdf"
DEFAULT = {"strategy": "heading", "size": 800, "overlap": 100}  # the Explorer's default sliders
WEDDING = "How many days off for my wedding?"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parsed(server_url):
    with open(SAMPLE_PDF, "rb") as f:
        r = requests.post(f"{server_url}/explorer/parse", files={"file": ("hr_leave_policy.pdf", f, "application/pdf")})
    assert r.status_code == 200
    return r.json()


def chunk(url, doc_id, strategy, size, overlap):
    r = requests.post(f"{url}/explorer/chunk", json={"doc_id": doc_id, "strategy": strategy, "size": size, "overlap": overlap})
    assert r.status_code == 200
    return r.json()


@pytest.fixture(scope="module")
def embedded(server_url, parsed):
    r = requests.post(f"{server_url}/explorer/embed", json={"doc_id": parsed["doc_id"], **DEFAULT}, timeout=60)
    assert r.status_code == 200
    return r.json()


def query(url, collection, text, k=4):
    r = requests.post(f"{url}/explorer/query", json={"collection": collection, "query": text, "k": k})
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------
def test_ex_01_parse_sample_pdf(parsed):
    assert len(parsed["pages"]) == 6
    assert parsed["total_chars"] > 10_000
    assert "Casual Leave" in parsed["full_text"]


def test_ex_02_fixed_chunks_are_contiguous(server_url, parsed):
    chunks = chunk(server_url, parsed["doc_id"], "fixed", 500, 0)["chunks"]
    assert all(c["length"] <= 500 for c in chunks)
    assert all(chunks[i]["start"] == chunks[i - 1]["end"] for i in range(1, len(chunks)))


def test_ex_03_fixed_overlap(server_url, parsed):
    chunks = chunk(server_url, parsed["doc_id"], "fixed", 500, 100)["chunks"]
    assert all(chunks[i]["start"] == chunks[i - 1]["end"] - 100 for i in range(1, len(chunks)))
    assert all(c["overlap_with_prev"] > 0 for c in chunks[1:])


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "heading"])
def test_ex_04_offsets_are_honest(server_url, parsed, strategy):
    full = parsed["full_text"]
    for c in chunk(server_url, parsed["doc_id"], strategy, 500, 100)["chunks"]:
        assert full[c["start"]:c["end"]] == c["text"]


def test_ex_05_smaller_size_more_chunks(server_url, parsed):
    small = chunk(server_url, parsed["doc_id"], "recursive", 200, 50)["stats"]["count"]
    large = chunk(server_url, parsed["doc_id"], "recursive", 800, 50)["stats"]["count"]
    assert small > large


def test_ex_06_heading_strategy_sections(server_url, parsed):
    chunks = chunk(server_url, parsed["doc_id"], "heading", 500, 100)["chunks"]
    assert all(c["section"] for c in chunks)
    assert any(c["section"].startswith("6.") for c in chunks)


def test_ex_07_embed(server_url, parsed, embedded):
    expected = chunk(server_url, parsed["doc_id"], **DEFAULT)["stats"]["count"]
    assert embedded["dims"] == 384
    assert embedded["count"] == expected
    assert len(embedded["points"]) == embedded["count"]


def test_ex_08_collection_rows(server_url, embedded):
    body = requests.get(f"{server_url}/explorer/collection/{embedded['collection']}", params={"limit": 20, "offset": 0}).json()
    for key in ("ids", "documents", "metadatas", "embeddings"):
        assert body[key]
    assert "page" in body["metadatas"][0] and "section" in body["metadatas"][0]


def test_ex_09_wedding_query_finds_marriage_leave(server_url, embedded):
    top = query(server_url, embedded["collection"], WEDDING)["results"][0]
    assert "Marriage Leave" in top["text"]


def test_ex_10_off_topic_scores_lower(server_url, embedded):
    wedding = query(server_url, embedded["collection"], WEDDING)["results"][0]["score"]
    cafeteria = query(server_url, embedded["collection"], "What is the cafeteria menu?")["results"][0]["score"]
    assert cafeteria < wedding


def test_ex_11_explorer_does_not_touch_chat_index(server_url, parsed):
    before = requests.get(f"{server_url}/health").json()["indexed_chunks"]
    r = requests.post(f"{server_url}/explorer/embed", json={"doc_id": parsed["doc_id"], "strategy": "fixed", "size": 300, "overlap": 0})
    assert r.status_code == 200
    assert requests.get(f"{server_url}/health").json()["indexed_chunks"] == before
    assert requests.delete(f"{server_url}/explorer/collection/{r.json()['collection']}").json() == {"deleted": True}


def _blank_pdf() -> bytes:
    from pypdf import PdfWriter

    writer, buf = PdfWriter(), io.BytesIO()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    return buf.getvalue()


PNG_1X1 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


@pytest.mark.parametrize("name,data", [("image.pdf", PNG_1X1), ("empty.pdf", _blank_pdf())], ids=["png-renamed", "blank-pdf"])
def test_ex_12_bad_pdf_is_friendly_400(server_url, name, data):
    r = requests.post(f"{server_url}/explorer/parse", files={"file": (name, data, "application/pdf")})
    assert r.status_code == 400
    assert r.json()["detail"]


# ---------------------------------------------------------------------------
# Browser tests
# ---------------------------------------------------------------------------
@pytest.fixture
def explorer(page):
    page.goto("/explorer")
    return page


@pytest.fixture
def sample_loaded(explorer):
    explorer.click(tid("ex-sample-button"))
    expect(explorer.locator(tid("ex-chunk-card")).first).to_be_visible()
    return explorer


def chunk_count(page) -> int:
    return int(page.text_content(tid("ex-chunk-count")))


def set_slider(page, name, value):
    """Move a slider and wait until the debounced re-chunk has updated the count."""
    before = page.text_content(tid("ex-chunk-count"))
    page.fill(tid(name), str(value))
    page.wait_for_function("prev => document.querySelector('[data-testid=ex-chunk-count]').textContent !== prev", arg=before)
    page.wait_for_timeout(300)


def test_ui_ex_01_sample_pdf(sample_loaded):
    expect(sample_loaded.locator(tid("ex-page-count"))).to_have_text("6")
    expect(sample_loaded.locator(tid("ex-raw-text"))).to_be_visible()
    expect(sample_loaded.locator(tid("ex-raw-text"))).to_contain_text("Nexora")


def test_ui_ex_02_chunk_view_renders(sample_loaded):
    n = chunk_count(sample_loaded)
    assert n > 0
    expect(sample_loaded.locator(tid("ex-chunk-span"))).to_have_count(n)
    expect(sample_loaded.locator(tid("ex-chunk-card"))).to_have_count(n)


def test_ui_ex_03_slider_updates_live(sample_loaded):
    start = chunk_count(sample_loaded)
    set_slider(sample_loaded, "ex-size-slider", 200)
    small = chunk_count(sample_loaded)
    assert small > start
    set_slider(sample_loaded, "ex-size-slider", 1200)
    assert chunk_count(sample_loaded) < small


def test_ui_ex_04_overlap_highlighting(sample_loaded):
    sample_loaded.click(tid("ex-strategy-fixed"))
    expect(sample_loaded.locator(tid("ex-overlap-span")).first).to_be_visible()
    sample_loaded.fill(tid("ex-overlap-slider"), "0")
    expect(sample_loaded.locator(tid("ex-overlap-span"))).to_have_count(0)
    sample_loaded.fill(tid("ex-overlap-slider"), "100")
    expect(sample_loaded.locator(tid("ex-overlap-span")).first).to_be_visible()


def test_ui_ex_05_strategy_switch(sample_loaded):
    sample_loaded.click(tid("ex-strategy-fixed"))
    expect(sample_loaded.locator(tid("ex-warning"), has_text="mid").first).to_be_visible()
    fixed_warnings = sample_loaded.locator(tid("ex-warning")).count()
    sample_loaded.click(tid("ex-strategy-heading"))
    expect(sample_loaded.locator(tid("ex-warning"))).not_to_have_count(fixed_warnings)
    assert sample_loaded.locator(tid("ex-warning")).count() < fixed_warnings


def test_ui_ex_06_embed_and_store(sample_loaded):
    n = chunk_count(sample_loaded)
    sample_loaded.click(tid("ex-embed-button"))
    expect(sample_loaded.locator(tid("ex-vector-row"))).to_have_count(n, timeout=30_000)
    expect(sample_loaded.locator(tid("ex-db-row")).first).to_be_visible()
    expect(sample_loaded.locator(tid("ex-scatter"))).to_be_visible()


def test_ui_ex_07_query_playground(sample_loaded):
    sample_loaded.fill(tid("ex-query-input"), WEDDING)
    sample_loaded.click(tid("ex-search-button"))
    first = sample_loaded.locator(f'{tid("ex-result-card")}[data-rank="1"]')
    expect(first).to_contain_text("Marriage", timeout=30_000)
    prompt = sample_loaded.locator(tid("ex-prompt-preview"))
    expect(prompt).to_contain_text(WEDDING)
    expect(prompt).to_contain_text("Marriage Leave")


def test_ui_ex_08_compare_mode(sample_loaded):
    sample_loaded.check(tid("ex-compare-toggle"))
    sample_loaded.fill(tid("ex-compare-size"), "200")          # main slider stays at 800
    sample_loaded.fill(tid("ex-query-input"), "How many Earned Leave days can I carry forward and encash?")
    sample_loaded.click(tid("ex-search-button"))
    k = int(sample_loaded.input_value(tid("ex-k-select")))
    expect(sample_loaded.locator(f'{tid("ex-compare-left")} {tid("ex-result-card")}')).to_have_count(k, timeout=30_000)
    expect(sample_loaded.locator(f'{tid("ex-compare-right")} {tid("ex-result-card")}')).to_have_count(k)


def test_ui_ex_09_upload_via_file_input(explorer):
    explorer.set_input_files(tid("ex-upload-input"), str(FIXTURES / "wfh_policy.md"))
    expect(explorer.locator(tid("ex-raw-text"))).to_contain_text("Work-From-Home")


def test_ui_ex_10_xss_safety(explorer):
    explorer.set_input_files(tid("ex-upload-input"), str(FIXTURES / "xss.md"))
    expect(explorer.locator(tid("ex-raw-text"))).to_contain_text("XSS test document")
    expect(explorer.locator(tid("ex-chunk-card")).first).to_be_visible()
    explorer.click(tid("ex-embed-button"))
    expect(explorer.locator(tid("ex-db-row")).first).to_be_visible()
    explorer.fill(tid("ex-query-input"), "hover me")
    explorer.click(tid("ex-search-button"))
    expect(explorer.locator(tid("ex-result-card")).first).to_be_visible()
    explorer.locator(tid("ex-chunk-card")).first.hover()
    assert explorer.evaluate("window.__xss") is None


def test_ui_ex_11_navigation(chat_page, base_url):
    chat_page.click(tid("explorer-link"))
    chat_page.wait_for_url(f"{base_url}/explorer")
    chat_page.click(tid("ex-back-link"))
    chat_page.wait_for_url(f"{base_url}/")
