from app.agents.query_resolution_agent import QueryResolutionAgent
from app.agents.data_extraction_agent import DataExtractionAgent
from app.agents.validation_agent import ValidationAgent


class RetailInsightsOrchestrator:
    """
    Orchestrator connecting 3 agents:
    - QueryResolutionAgent: natural language -> intent (mode + normalized_query)
    - DataExtractionAgent: RAG retrieval from vector store
    - ValidationAgent: final answer / summary generation
    """

    def __init__(self, folder_path: str):
        self.query_agent = QueryResolutionAgent()
        self.data_agent = DataExtractionAgent(folder_path=folder_path)
        self.validation_agent = ValidationAgent()

    def summarization_mode(self) -> str:
        intent = {
            "mode": "summarization",
            "normalized_query": "Provide an overall summary of sales performance and key insights.",
            "notes": "System-generated summarization query",
        }
        docs = self.data_agent.retrieve_context(intent["normalized_query"])
        return self.validation_agent.answer("summarization", intent["normalized_query"], intent, docs)

    def conversational_qa(self, user_query: str) -> str:
        intent = self.query_agent.plan(user_query)
        mode = intent.get("mode", "qa")
        q_for_retrieval = intent.get("normalized_query", user_query)

        docs = self.data_agent.retrieve_context(q_for_retrieval)
        return self.validation_agent.answer(mode, user_query, intent, docs)
