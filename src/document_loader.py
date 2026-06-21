from pypdf import PdfReader
from docx import Document


def extract_text(file):

    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".pdf"):

        pdf = PdfReader(file)

        text = ""

        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        return text

    elif file.name.endswith(".docx"):

        doc = Document(file)

        text = ""

        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text

    return ""