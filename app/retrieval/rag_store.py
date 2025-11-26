from pathlib import Path
from typing import List
import json

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()
class RetailRAGStore:
    """
    - Ingests files from a folder:
    - For CSV / Excel:
        * Creates 1 summary document per file/sheet
        * PLUS multiple chunk documents with actual row data
          (so the full content is searchable via RAG)

    - Uses SentenceTransformer embeddings (local)
    - Stores everything in a persistent Chroma vector DB on disk
      so ingestion happens only once, and subsequent runs just load it.
    """

    def __init__(self, persist_directory: str = "vector_db"):
        # Folder where Chroma will store its index
        self.persist_directory = persist_directory
 
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
 
        self.vectorstore = None
    #preparing csv files into documents
    def _csv_to_docs(self, path: Path) -> List[Document]:
        df = pd.read_csv(path, low_memory=False)
        docs: List[Document] = []

        num_rows, num_cols = df.shape
        col_names = list(df.columns)
        summary_lines = [
            f"File {path.name}:",
            f"- Rows: {num_rows}",
            f"- Columns ({min(len(col_names), 20)} of {len(col_names)}): "
            + ", ".join(col_names[:20]),
        ]

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            summary_lines.append(f"- Numeric columns: {', '.join(numeric_cols[:10])}")
            try:
                desc = df[numeric_cols].describe().round(2).to_string()
                summary_lines.append("- Basic numeric statistics:\n" + desc)
            except Exception:
                pass

        summary_text = "\n".join(summary_lines)

        docs.append(
            Document(
                page_content=summary_text,
                metadata={
                    "source": str(path),
                    "type": "csv_summary",
                },
            )
        )
        chunk_size = 200  # tune between 100–500 if needed

        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk_df = df.iloc[start:end]

            # Tabular representation of this chunk (keeps headers)
            try:
                table_text = chunk_df.to_markdown(index=False)
            except Exception:
                table_text = chunk_df.to_csv(index=False)

            chunk_text = (
                f"File {path.name} - rows {start} to {end - 1}:\n"
                f"{table_text}"
            )

            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": str(path),
                        "type": "csv_chunk",
                        "start_row": start,
                        "end_row": end - 1,
                    },
                )
            )

        return docs
#preparing docs for excel files
    def _excel_to_docs(self, path: Path) -> List[Document]:
        xls = pd.ExcelFile(path)
        docs: List[Document] = []

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            num_rows, num_cols = df.shape
            col_names = list(df.columns)

            summary_text = (
                f"File {path.name}, Sheet {sheet}:\n"
                f"- Rows: {num_rows}\n"
                f"- Columns ({min(len(col_names), 20)} of {len(col_names)}): "
                + ", ".join(col_names[:20])
            )

            docs.append(
                Document(
                    page_content=summary_text,
                    metadata={
                        "source": str(path),
                        "sheet": sheet,
                        "type": "excel_sheet_summary",
                    },
                )
            )

            chunk_size = 200  # same idea as CSV

            for start in range(0, num_rows, chunk_size):
                end = min(start + chunk_size, num_rows)
                chunk_df = df.iloc[start:end]

                try:
                    table_text = chunk_df.to_markdown(index=False)
                except Exception:
                    table_text = chunk_df.to_csv(index=False)

                chunk_text = (
                    f"File {path.name}, Sheet {sheet} - rows {start} to {end - 1}:\n"
                    f"{table_text}"
                )

                docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": str(path),
                            "sheet": sheet,
                            "type": "excel_sheet_chunk",
                            "start_row": start,
                            "end_row": end - 1,
                        },
                    )
                )

        return docs
#preparing text files into documents
    def _txt_to_docs(self, path: Path) -> List[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Truncate to keep prompts manageable
        return [
            Document(
                page_content=f"Text report from {path.name}:\n{text[:3000]}",
                metadata={"source": str(path), "type": "txt"},
            )
        ]
#preparing json files into documents
    def _json_to_docs(self, path: Path) -> List[Document]:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        text = json.dumps(obj, indent=2)
        return [
            Document(
                page_content=f"JSON report from {path.name}:\n{text[:3000]}",
                metadata={"source": str(path), "type": "json"},
            )
        ]

    def _process_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._csv_to_docs(path)
        elif suffix in [".xlsx", ".xls"]:
            return self._excel_to_docs(path)
        elif suffix == ".txt":
            return self._txt_to_docs(path)
        elif suffix == ".json":
            return self._json_to_docs(path)
        return []

    #main ingestion function
    def ingest_folder(self, folder_path: str):
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"{folder_path} is not a valid folder")
#vector db already exists
        if self.persist_directory and Path(self.persist_directory).exists():
            print(f"[RAG] Loading existing Chroma DB from {self.persist_directory}")
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="retail_docs",
            )
            return  # Skip ingestion
#vector db does not exist, creating new one
        print("[RAG] Ingesting all files and creating new Chroma DB...")
        all_docs: List[Document] = []
        exts = [".csv", ".xlsx", ".xls", ".txt", ".json"]

        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue

            try:
                docs = self._process_file(path)
                all_docs.extend(docs)
            except Exception as e:
                
                print(f"[RAG] Skipping {path} due to error: {e}")

        
        self.vectorstore = Chroma.from_documents(
            all_docs,
            embedding=self.embeddings,
            collection_name="retail_docs",
            persist_directory=self.persist_directory,
        )
        print(f"[RAG] Ingestion done. Saved to disk at {self.persist_directory}.")

    #retriever function
    def as_retriever(self, k: int = 8):
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore is not initialized. Call ingest_folder() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
