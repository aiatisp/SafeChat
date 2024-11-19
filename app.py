import os
from dotenv import load_dotenv
import gradio as gr
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant."
}

def respond(message, history):
    # Initialize chat history with system prompt
    chat_messages = [SYSTEM_PROMPT]
    
    # Add conversation history
    for human, assistant in history:
        chat_messages.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    
    # Add current message
    chat_messages.append({"role": "user", "content": message})
    
    # Get response from Groq
    response = client.chat.completions.create(
	      model=“gemma2-9b-it”,
        messages=chat_messages,
        max_tokens=100,
        temperature=1.2
    )
    
    return response.choices[0].message.content

# Create Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title=“ISP 🐬 AI Club 🤖 Project #1: Creating a Basic Chatbot",
    description="Chat with a Groq-powered LLM assistant",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)
