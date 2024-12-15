import streamlit as st
from functions import completed, err, warn, progress_bar

st.title("Embedding Generator")

uploaded_file = st.file_uploader("Upload a single PDF file", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    # Check the uploaded file type
    if uploaded_file.type == "application/pdf":
        progress_bar("Processing uploaded file...")
        completed(f"File name: {uploaded_file.name} uploaded!", 2)
    else:
        err("Invalid file type uploaded. Please upload a PDF file.", 3)
