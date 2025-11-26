Retail businesses generate large amounts of sales data in formats such as CSV, Excel, JSON & TXT reports.
This project builds a Retail Insights Assistant — a GenAI + RAG-based application that:
Ingests reports & sales data
Stores them in a vector database (Chroma)
Uses Azure OpenAI (GPT + Embeddings)
Supports two modes:

Summarization Mode → Generates automatic business insights

Conversational Q&A Mode → Answers natural language questions 

Uses multi-agent architecture (LangChain-based):
Agents-Role
Query Resolution Agent -Understand user query & decide mode
Data Extraction Agent -	Retrieve relevant chunks using RAG
Validation Agent -	Generate final human-readable answer

Project Structure
retail_insights_assistant/
│
├── app/
│   ├── ui/
│   │    └── streamlit_app.py        # Streamlit UI
│   ├── agents/
│   │    ├── query_resolution_agent.py
│   │    ├── data_extraction_agent.py
│   │    └── validation_agent.py
│   ├── retrieval/
│   │    └── rag_store.py            # Vector DB + RAG logic             
│   └── orchestrator.py              # Multi-agent pipeline
│
├── data/                            # CSV, Excel, Reports go here
├── vector_db/                       # Auto-created after first ingestion
├── requirements.txt
├── README.md
├── .env                             # Azure OpenAI credentials
└── venv/                            # Virtual environment

Setup Instructions:
1.Create Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows

2.Install Dependencies
pip install -r requirements.txt

3.Add Your Azure OpenAI Credentials (.env)

4.Ensure Dataset Exists
Place all CSV / Excel / JSON / TXT files inside data/ folder.

5.Run the Application
streamlit run app/ui/streamlit_app.py
