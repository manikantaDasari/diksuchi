"""
python_langchain.py
────────────────────
Connect LangChain to your local Diksuchi router.
pip install langchain langchain-openai
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ROUTER_BASE = "http://localhost:8080/v1"

# ── 1. Swap in the router as your LLM ────────────────────────────────────────
llm = ChatOpenAI(
    base_url=ROUTER_BASE,
    api_key="router",
    model="llama3.2",           # router may override based on rules
    temperature=0.7,
    streaming=False,
)

# Streaming variant
llm_stream = ChatOpenAI(
    base_url=ROUTER_BASE,
    api_key="router",
    model="llama3.2",
    streaming=True,
)


# ── 2. Simple invocation ──────────────────────────────────────────────────────
def simple_invoke():
    response = llm.invoke([HumanMessage(content="Explain decorators in Python.")])
    print("[simple_invoke]", response.content[:300], "\n")


# ── 3. Prompt template + chain (LCEL) ────────────────────────────────────────
def prompt_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {language} developer. Be concise."),
        ("human",  "{task}"),
    ])
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "language": "Python",
        "task":     "Write a decorator that measures function execution time.",
    })
    print("[prompt_chain]\n", result, "\n")


# ── 4. Streaming with LCEL ────────────────────────────────────────────────────
def streaming_chain():
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain  = prompt | llm_stream | StrOutputParser()

    print("[streaming_chain] ", end="")
    for chunk in chain.stream({"question": "List 5 Python best practices."}):
        print(chunk, end="", flush=True)
    print("\n")


# ── 5. Code generation pipeline ──────────────────────────────────────────────
from langchain_core.prompts import PromptTemplate

def code_pipeline(task: str) -> str:
    code_prompt = PromptTemplate.from_template(
        "Write clean, documented Python code for the following task. "
        "Return ONLY the code, no explanation.\n\nTask: {task}"
    )
    review_prompt = PromptTemplate.from_template(
        "Review this Python code for bugs and improvements:\n\n{code}\n\n"
        "List any issues found."
    )

    parser = StrOutputParser()
    generate_chain = code_prompt | llm | parser
    review_chain   = review_prompt | llm | parser

    code = generate_chain.invoke({"task": task})
    review = review_chain.invoke({"code": code})

    return f"### Generated Code:\n{code}\n\n### Review:\n{review}"


# ── 6. Router-aware LLM wrapper ──────────────────────────────────────────────
import httpx

class RouterAwareChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that logs routing decisions."""

    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config, **kwargs)
        # Fetch last routing decision from router log
        try:
            log = httpx.get("http://localhost:8080/v1/log?limit=1").json()
            if log["entries"]:
                e = log["entries"][0]
                print(f"  [router] {e['backend']}/{e['provider']} · rule={e['rule']} · {e['tokens']} tokens · {e['duration_ms']}ms")
        except Exception:
            pass
        return response


if __name__ == "__main__":
    simple_invoke()
    prompt_chain()
    streaming_chain()

    print("[code_pipeline]")
    result = code_pipeline("Read a JSON file and validate its schema using pydantic")
    print(result[:600])
