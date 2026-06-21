import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_answer(question, context):

    prompt = f"""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = model.generate_content(prompt)

    return response.text