"""
python_llamaindex.py
─────────────────────
Connect LlamaIndex to your local Diksuchi router.
pip install llama-index llama-index-llms-openai
"""

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage

ROUTER_BASE = "http://localhost:8080/v1"

# ── 1. Configure LlamaIndex to use the router ─────────────────────────────────
llm = OpenAI(
    api_base=ROUTER_BASE,
    api_key="router",
    model="llama3.2",
    temperature=0.1,
)
Settings.llm = llm    # set globally so all LlamaIndex pipelines use the router


# ── 2. Basic chat ─────────────────────────────────────────────────────────────
def basic_chat():
    response = llm.chat([
        ChatMessage(role="user", content="What is RAG in AI?"),
    ])
    print("[basic_chat]", response.message.content[:300], "\n")


# ── 3. Document Q&A (RAG pipeline) ────────────────────────────────────────────
def rag_example(docs_path: str = "./docs"):
    """
    Build a RAG pipeline over a local folder of documents.
    All LLM calls are routed through your local Diksuchi router.
    """
    import os
    if not os.path.exists(docs_path):
        print(f"[rag_example] docs_path {docs_path!r} not found — skipping.")
        return

    documents = SimpleDirectoryReader(docs_path).load_data()
    index     = VectorStoreIndex.from_documents(documents)
    engine    = index.as_query_engine()

    result = engine.query("What are the main topics in these documents?")
    print("[rag_example]", str(result)[:400], "\n")


# ── 4. Code assistant ─────────────────────────────────────────────────────────
from llama_index.core.prompts import PromptTemplate

CODE_TMPL = PromptTemplate(
    "You are an expert Python developer.\n"
    "Task: {task}\n"
    "Write clean, well-commented Python code. Return ONLY the code."
)

def code_assistant(task: str) -> str:
    prompt = CODE_TMPL.format(task=task)
    response = llm.complete(prompt)
    return response.text


if __name__ == "__main__":
    basic_chat()
    rag_example()

    print("[code_assistant]")
    code = code_assistant("Create a context manager that times a code block")
    print(code[:500])
