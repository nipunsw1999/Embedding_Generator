from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GEN_AI_KEY"))

def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="retrieval_document")
    return embedding['embedding']




def pdf_load(pdf, limit: int):
    """
    Load text from a PDF using PyPDFLoader with a page limit.
    :param pdf: File-like object or file path.
    :param limit: Number of pages to process.
    :return: Combined text from the specified number of pages.
    """
    try:
        # Save the BytesIO to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # Adjust limit to avoid exceeding total pages
        total_pages = len(pages)
        limit = min(limit, total_pages)

        # Extract text from the specified pages
        text = "\n".join([doc.page_content for doc in pages[:limit] if doc.page_content.strip()])

        return text

    except Exception as e:
        raise ValueError(f"Error loading PDF: {str(e)}")

