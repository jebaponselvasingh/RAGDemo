# Lab notebooks

The six lab steps as Jupyter notebooks, to run cell by cell in class. They import the lab's own
code (`rag/`, `agent/`, `llm.py`), so they behave exactly like the `steps/*.py` scripts.

| Notebook | Topic |
|---|---|
| `step1_embeddings.ipynb` | Sentences to vectors; meaning vs keyword search |
| `step2_rag.ipynb` | Load, chunk, embed, store, retrieve, generate; with vs without RAG |
| `step3_agent_scratch.ipynb` | A ReAct agent in plain Python |
| `step4_agent_langgraph.ipynb` | The same agent with LangGraph |
| `step5_evaluate.ipynb` | Scoring the agent (tool accuracy, source hit, facts, LLM-as-judge) |
| `step6_memory.ipynb` | Follow-up questions: no memory, memory by hand, checkpointer |

## Run them

From the `day5-agentic-rag-lab` folder, with Ollama running (`ollama pull llama3.2:3b`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-notebooks.txt
cp .env.example .env               # Windows: copy .env.example .env
jupyter lab notebooks/
```

Select the kernel from your `.venv`, then run the setup cell at the top of each notebook first.
Step 1 runs without Ollama; the others need it for the LLM cells.
