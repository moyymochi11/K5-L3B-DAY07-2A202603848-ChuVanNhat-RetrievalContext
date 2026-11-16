"""LLM Controller for Agentic RAG using OpenAI gpt-4o-mini."""
import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are an accurate, grounded AI Agent answering questions about the VinUniversity International Student Handbook.
You have access to a search tool over the handbook.

Format of context chunks:
[chunk_id | source | page X]
Text of the chunk...

Rules:
1. When you need to retrieve information, return a search action:
   {"action": "search", "query": "<focused search keywords>"}

2. When the context contains sufficient evidence to answer the user's question, answer factually and strictly based on the context.
   You MUST cite the supporting chunk ID in square brackets like [chunk_id] directly in your answer text, AND include it in the citations list:
   {"action": "answer", "answer": "The campus area is 420 hectares [vinuni-handbook-2024_p5_para_0].", "citations": ["vinuni-handbook-2024_p5_para_0"]}

3. If the context DOES NOT contain enough evidence:
   - If you think searching with different keywords (e.g. specific department, section, or synonym) will find the evidence, return a new search action:
     {"action": "search", "query": "<new keywords>"}
   - If the handbook clearly does not contain information to answer the question (e.g. topics out of scope like pets in dorms, master's tuition fees, specific airline flight discounts, exchange application deadlines), OR if multiple searches yielded no relevant facts, DO NOT hallucinate. Return:
     {"action": "answer", "answer": "The handbook does not provide sufficient evidence to answer this question.", "citations": []}

4. Output ONLY a valid JSON object matching either {"action": "search", ...} or {"action": "answer", ...}. Do not include markdown code block formatting or extraneous text."""


class LLMController:
    """Agent controller using real LLM (gpt-4o-mini)."""
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, max_retries: int = 3):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or .env")
        self.client = OpenAI(api_key=self.api_key)
        self.max_retries = max_retries
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Examine question and history; decide to search or answer."""
        # Initial turn: search directly with the question
        if not history:
            return {"action": "search", "query": question}

        # Build messages from history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add user query and context
        user_prompt = f"Question: {question}\n\nSearch History & Retrieved Context:\n"
        for idx, step in enumerate(history):
            queries = ", ".join(step.get("queries", []))
            context = step.get("context", "")
            context_ids = step.get("context_ids", [])
            user_prompt += f"--- Search Queries: {queries} ---\nAvailable Chunk IDs: {context_ids}\nContext:\n{context}\n\n"

        user_prompt += "Decide your next action. Return ONLY JSON."
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                usage = response.usage
                if usage:
                    self.total_prompt_tokens += usage.prompt_tokens
                    self.total_completion_tokens += usage.completion_tokens

                content = response.choices[0].message.content or "{}"
                data = json.loads(content)

                action = data.get("action")
                if action == "search":
                    query = data.get("query", "").strip()
                    if query:
                        return {"action": "search", "query": query}
                elif action == "answer":
                    answer = data.get("answer", "").strip()
                    citations = data.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []
                    return {"action": "answer", "answer": answer, "citations": citations}

                # If format unexpected, default to search or abstain based on searches
                return {"action": "search", "query": question}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM Controller failed after {self.max_retries} attempts: {e}")
                time.sleep(1.0)

        return {"action": "answer", "answer": "The handbook does not provide sufficient evidence to answer this question.", "citations": []}

    @property
    def cost_usd(self) -> float:
        """Calculate estimated cost for gpt-4o-mini."""
        # gpt-4o-mini: $0.15 / 1M prompt, $0.60 / 1M completion
        return (self.total_prompt_tokens / 1_000_000) * 0.15 + (self.total_completion_tokens / 1_000_000) * 0.60
