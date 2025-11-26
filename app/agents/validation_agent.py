from typing import List
import os
 
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.schema import Document
 
load_dotenv()
 
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
 
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
 
 
VALIDATION_SYSTEM_PROMPT = """
You are the Validation & Answering Agent for a Retail RAG Assistant.
 
You receive:
- mode: "qa" or "summarization"
- the original user question
- a JSON intent from the Query Resolution Agent
- retrieved context documents (from CSV/Excel/TXT/JSON-based summaries and chunks)
 
Your tasks:
1. If mode is "summarization":
   - Provide an executive-level summary of overall performance and key insights.
   - Mention regions, categories, products, or time periods if you can infer them.
   - Use short paragraphs and/or bullet points. No raw tables.
 
2. If mode is "qa":
   - Answer the question using ONLY the retrieved context.
   - DO NOT hallucinate facts not supported by the context.
   - DO NOT dump raw tables or CSV-like content.
   - Summarize patterns and key points in plain business language.
   - If you cannot find enough information, clearly say so.
 
3. Always keep answers business-friendly, concise, and clear.
"""
 
 
class ValidationAgent:
    """
    Uses Azure OpenAI Chat Completion to generate the final answer/summary,
    grounded in RAG-retrieved context.
    """
 
    def __init__(self, deployment_name: str | None = None):
        self.deployment = deployment_name or AZURE_DEPLOYMENT
 
    def answer(self, mode: str, user_query: str, intent: dict, docs: List[Document]) -> str:
        # 1) Limit docs (we don't want to send huge prompts)
        max_docs = 5
        trimmed_docs = docs[:max_docs] if docs else []
 
        # 2) Truncate each doc's content
        per_doc_char_limit = 1500
        trimmed_contents = []
        for d in trimmed_docs:
            content = d.page_content or ""
            if len(content) > per_doc_char_limit:
                content = content[:per_doc_char_limit]
            trimmed_contents.append(content)
 
        context_text = "\n\n---\n\n".join(trimmed_contents) if trimmed_contents else "No relevant context found."
 
        # 3) Global cap to stay safely within context window
        global_char_limit = 6000
        if len(context_text) > global_char_limit:
            context_text = context_text[:global_char_limit]
 
        user_prompt = f"""
Mode: {mode}
User question:
{user_query}
 
Intent (from Query Resolution Agent):
{intent}
 
Retrieved context (possibly truncated):
{context_text}
 
Now produce the final answer according to the system instructions.
"""
 
        resp = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
 
        answer = resp.choices[0].message.content or ""
        return answer.strip()