from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GEN_AI_KEY"))

def get_embeddings(text):
    """
    Generate embeddings for a given text using Google Generative AI.

    :param text: The input text to be embedded.
    :return: The embedding vector.
    :raises: Exception if embedding generation fails.
    """
    try:
        model = 'models/embedding-001'
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return embedding['embedding']
    except Exception as e:
        raise ValueError(f"Error generating embeddings: {str(e)}")


def pdf_load(pdf, limit: int):
    """
    Load text from a PDF file with a page limit using PyPDFLoader.

    :param pdf: File-like object or file path.
    :param limit: Number of pages to process.
    :return: Combined text from the specified number of pages.
    :raises: ValueError if PDF loading or text extraction fails.
    """
    try:
        # Save the uploaded PDF to a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        # Load and split the PDF into pages using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # Adjust limit to avoid exceeding total number of pages
        total_pages = len(pages)
        limit = min(limit, total_pages)

        # Extract text from the specified number of pages
        text = "\n".join([
            doc.page_content for doc in pages[:limit] if doc.page_content.strip()
        ])

        return text

    except Exception as e:
        raise ValueError(f"Error loading PDF: {str(e)}")
