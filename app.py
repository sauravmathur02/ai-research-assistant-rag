import streamlit as st

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
        st.write(f"📄 {file.name}")