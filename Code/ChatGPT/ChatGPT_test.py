import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Your OpenAI API key
api_key=os.getenv('OPENAI_API_KEY')

# Function to generate responses using GPT-3.5 turbo model
def generate_response(question):
  # Define the context containing the bank of phrases. Need to store each relation twice to allow for querying in both directions
  context = """
  chair left of couch.
  couch left of lamp
  lamp on desk.
  book under table.
  """
  openai.api_key = api_key
  prompt = f"{question}\nContext: {context}"
  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=100
  )
  return response.choices[0].text

# Main function for user interaction
def main():
    print("Welcome to the Object Position Query System")
    while True:
        user_input = input("Ask a question (e.g., 'Where is the chair in relation to the couch?') or type 'exit' to quit: ")

        if user_input.lower() == 'exit':
            break

        # Use the user's question as a prompt to the model
        prompt = user_input.strip() + "\n"
        response = generate_response(prompt)

        print("Response: " + response)

if __name__ == "__main__":
    main()
