import streamlit as st
from functions import completed, err, warn, progress_bar

st.title("Embedding Generator")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Simulate a loading progress bar for the file upload
    st.success("PDF uploaded successfully!")
    progress_bar("Processing uploaded file...")
    completed(f"File name: {uploaded_file.name}", 2)