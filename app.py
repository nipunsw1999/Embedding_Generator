import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import pandas as pd
import os
from google.generativeai import configure, embed_content

# Load environment variables
load_dotenv()

# Configure API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
configure(api_key=os.getenv("GEN_AI_KEY"))

# Function to generate embeddings using Google Generative AI
def get_embeddings(text):
    try:
        model = 'models/embedding-001'
        embedding = embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return embedding['embedding']
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Function to load text from a PDF with a limit
def pdf_load(pdf, limit: int):
    """
    Load text from a PDF using PyPDFLoader with a page limit.
    :param pdf: File-like object or file path.
    :param limit: Number of pages to process.
    :return: Combined text from the specified number of pages.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        total_pages = len(pages)
        limit = min(limit, total_pages)

        text = "\n".join([doc.page_content for doc in pages[:limit] if doc.page_content.strip()])
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

# Sidebar menu
with st.sidebar:
    selected = option_menu("Menu", ["Generator", "About"],
                           icons=["house", "info-circle"], 
                           menu_icon="cast", 
                           default_index=0)

# Main content rendering based on menu selection
if selected == "Generator":
    st.title(":green[Embedding Generator]")

    # User input for chunking parameters
    chunk_size = st.slider("Select Chunk Size:", min_value=100, max_value=1000, value=500, step=50)
    chunk_overlap = st.slider("Select Chunk Overlap:", min_value=50, max_value=500, value=150, step=25)

    # File upload
    uploaded_file = st.file_uploader("Upload a single PDF file", type=["pdf"], accept_multiple_files=False)
    start = st.button(":red[Start Embedding Generator]")
    
    if start:
        if uploaded_file:
            # Check if the uploaded file is a valid PDF
            if uploaded_file.type == "application/pdf":
                with st.spinner('Processing your file...'):
                    # Load text from the uploaded PDF
                    text = pdf_load(BytesIO(uploaded_file.read()), limit=4)
                    
                    if text:
                        # Split the text into chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            is_separator_regex=False,
                        )
                        
                        docs = text_splitter.create_documents([text])
                        for i, d in enumerate(docs):
                            d.metadata = {"doc_id": i}
                        
                        # Extract content and generate embeddings
                        content_list = [doc.page_content for doc in docs]
                        embeddings = []
                        
                        # Progress bar for embedding generation
                        progress = st.progress(0)
                        for i, content in enumerate(content_list):
                            embedding = get_embeddings(content)
                            if embedding is not None:
                                embeddings.append(embedding)
                            progress.progress((i + 1) / len(content_list))
                        
                        # Create DataFrame and CSV
                        if embeddings:
                            dataframe = pd.DataFrame({
                                'page_content': content_list,
                                'embeddings': embeddings
                            })
                            
                            csv_data = dataframe.to_csv(index=False)
                            st.success("Embeddings generated successfully!")
                            st.download_button(
                                label=":red[Download CSV]",
                                data=csv_data,
                                file_name=f"{uploaded_file.name[:-4]}_embeddings.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error("Failed to generate embeddings. Please try again.")
                    else:
                        st.error("Failed to extract text from the uploaded PDF.")
            else:
                st.error("Invalid file type uploaded. Please upload a valid PDF file.")
        else:
            st.warning("Please upload a PDF file to proceed.")

elif selected == "About":
    st.title("About")
    st.write("""
        This is the Embedding Generator application. It uses advanced natural language processing techniques
        to extract and process embeddings from PDF documents. For more information, visit our documentation or
        contact the developer.
    """)
    st.write("Built with Streamlit and LangChain.")
