from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found!")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize correct model (no "models/" prefix)
model = genai.GenerativeModel('gemini-pro')

def chat_with_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    print("Welcome to the Gemini Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input.strip():
            print("Please enter something!")
            continue
        response = chat_with_gemini(user_input)
        print("Gemini:", response)

if __name__ == "__main__":
    main()
