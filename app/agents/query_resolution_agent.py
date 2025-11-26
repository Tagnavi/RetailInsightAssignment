import json
import os
 
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv()
 
# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
 
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
 
 
QUERY_RESOLUTION_SYSTEM = """
You are the Query Resolution Agent for a Retail RAG Assistant.
 
Given a user query, decide:
- mode: "qa" (conversational question answering) or "summarization"
- normalized_query: a cleaned-up query suitable for retrieval
- notes: optional hints (like region, category, time period).
 
Return STRICT JSON only. Example:
{
  "mode": "qa",
  "normalized_query": "sales performance of north region in Q4",
  "notes": "focus on underperforming categories"
}
 
DO NOT add explanations.
DO NOT wrap JSON in markdown or ```json.
"""
 
 
class QueryResolutionAgent:
    """
    Uses Azure OpenAI Chat Completion to convert user query into
    a small JSON intent object:
      - mode: qa / summarization
      - normalized_query: query for RAG
      - notes: optional hints
    """
 
    def __init__(self, deployment_name: str | None = None):
        self.deployment = deployment_name or AZURE_DEPLOYMENT
 
    def plan(self, user_query: str) -> dict:
        messages = [
            {"role": "system", "content": QUERY_RESOLUTION_SYSTEM},
            {
                "role": "user",
                "content": f"User query:\n{user_query}\n\nRespond with JSON only.",
            },
        ]
 
        resp = client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0,
        )
 
        raw = resp.choices[0].message.content or ""
 
        try:
            # Clean any accidental markdown wrapping
            cleaned = raw.strip().strip("```json").strip("```").strip()
            return json.loads(cleaned)
        except Exception:
            # Fallback – just treat it as a QA query
            return {
                "mode": "qa",
                "normalized_query": user_query,
                "notes": f"Could not parse JSON from LLM. Raw: {raw[:200]}",
            }