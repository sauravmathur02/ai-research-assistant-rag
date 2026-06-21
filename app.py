import streamlit as st
from src.document_loader import extract_text
from src.chunker import chunk_text

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded")

    for file in uploaded_files:

        st.subheader(f"📄 {file.name}")

        text = extract_text(file)
        chunks = chunk_text(text)
        st.write(f"Total Chunks: {len(chunks)}")

        st.text_area(
            "Document Preview",
            text[:5000],
            height=300
        )