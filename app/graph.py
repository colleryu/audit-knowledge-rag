from typing import TypedDict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.config import settings
from app.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from app.retriever import search_audit_knowledge, format_docs_as_context


class RAGState(TypedDict):
    question: str
    docs: list[dict[str, Any]]
    context: str
    answer: str


llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    temperature=0.2,
)


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]

    docs = search_audit_knowledge(question)
    context = format_docs_as_context(docs)

    return {
        **state,
        "docs": docs,
        "context": context,
    }


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    docs = state["docs"]

    if not docs or docs[0].get("score", 0) < 0.6:
        return {
            **state,
            "answer": "当前知识库中未检索到明确依据。",
        }

    context = state["context"]

    user_prompt = RAG_USER_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(
        [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    return {
        **state,
        "answer": response.content,
    }

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()


def ask_audit_rag(question: str) -> dict[str, Any]:
    result = rag_graph.invoke(
        {
            "question": question,
            "docs": [],
            "context": "",
            "answer": "",
        }
    )

    return {
        "question": question,
        "answer": result["answer"],
        "sources": [
            {
                "score": doc.get("score"),
                "title": doc.get("title"),
                "publish_date": doc.get("publish_date"),
                "source": doc.get("source"),
                "url": doc.get("url"),
            }
            for doc in result["docs"]
        ],
    }
