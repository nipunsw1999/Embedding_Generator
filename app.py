import time
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from functions import completed, err, warn, progress_bar
from functions_emb import *
from io import BytesIO
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GEN_AI_KEY"))

with st.sidebar:
    selected = option_menu("Menu", ["Generator", "About"], 
        icons=["house", "info-circle"], menu_icon="cast", default_index=0)

# Main content rendering based on menu selection
if selected == "Generator":
    st.title(":green[Embedding Generator]")

    uploaded_file = st.file_uploader("Upload a single PDF file", type=["pdf"], accept_multiple_files=False)
    start = st.button(":red[Start Embedding Generator]")
    if start:
        if uploaded_file:
            # Check the uploaded file type
            if uploaded_file.type == "application/pdf":
                
                with st.spinner(f'Wait for it...'):
                    # Wrap uploaded_file to BytesIO for compatibility with PyPDFLoader
                    text = pdf_load(BytesIO(uploaded_file.read()), 4)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=150,
                        length_function=len,
                        is_separator_regex=False,
                    )
                    
                    docs = text_splitter.create_documents([text])
                    for i, d in enumerate(docs):
                        d.metadata = {"doc_id": i}
                        
                    # Get the page_content from the documents and create a new list
                    content_list = [doc.page_content for doc in docs]
                    # Send one page_content at a time
                    embeddings = [get_embeddings(content) for content in content_list]
                    
                    # Create a dataframe to ingest it to the database
                    dataframe = pd.DataFrame({
                        'page_content': content_list,
                        'embeddings': embeddings
                    })
                    
                    csv_data = dataframe.to_csv(index=False)
                    st.toast("Ready to download", icon="🌟")
                    st.snow()
                    name = uploaded_file.name[:-4]
                    st.download_button(
                            label=":red[Download]",
                            data=csv_data,
                            file_name=f"{name} embeddings.csv",
                            mime="text/csv",
                    )
            else:
                err("Invalid file type uploaded. Please upload a PDF file.", 3)
        else:
            warn("Please upload a PDF file", 2)

elif selected == "About":
    st.title("About")
    st.write("""
        This is the Embedding Generator application. It uses advanced natural language processing techniques
        to extract and process embeddings from PDF documents. For more information, visit our documentation or
        contact the developer.
    """)
    st.write("Built with Streamlit and LangChain.")

