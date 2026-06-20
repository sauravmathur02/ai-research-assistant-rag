import streamlit as st
from pypdf import PdfReader
from docx import Document

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

        if file.name.endswith(".txt"):

            content = file.read().decode("utf-8")

            st.text_area(
                "File Content",
                content,
                height=200
            )

        elif file.name.endswith(".pdf"):

            pdf = PdfReader(file)

            text = ""

            for page in pdf.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

            st.text_area(
                "PDF Content Preview",
                text[:5000],
                height=300
            )
        elif file.name.endswith(".docx"):

            doc = Document(file)

            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            st.text_area(
                "DOCX Content Preview",
                text[:5000],
                height=300
            )
        else:
                st.info("Preview not implemented yet.")        