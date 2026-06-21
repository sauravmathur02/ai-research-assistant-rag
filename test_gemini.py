from src.llm import model

print("Gemini loaded successfully")

response = model.generate_content(
    "What is Generative AI?"
)

print("Response received")
print(response.text)