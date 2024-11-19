import os
from dotenv import load_dotenv
import gradio as gr
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# System prompts
GUARD_PROMPT = {
    "role": "system",
    "content": """You are a content safety system. Evaluate if the user's input is safe and appropriate.
    Respond only with "SAFE" or "UNSAFE". Consider the following unsafe:
    - Inappropriate language
    - Harmful or dangerous content
    - Hate speech or discrimination
    - Personal information requests
    - Explicit or adult content
    - Violence or threats
    - Illegal activities
    """
}

CHAT_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant."
}

def check_safety(message):
    # Get safety check from Llama Guard
    guard_messages = [
        GUARD_PROMPT,
        {"role": "user", "content": message}
    ]
    
    safety_response = client.chat.completions.create(
        model="llama-guard-3-8b",
        messages=guard_messages,
        max_tokens=10,
        temperature=0.1
    )
    
    return safety_response.choices[0].message.content.strip().upper() == "SAFE"

def respond(message, history):
    # First check if content is safe
    if not check_safety(message):
        return "I apologize, but I cannot help with that request as it was classified as potentially inappropriate or unsafe."
    
    # Initialize chat history with system prompt
    chat_messages = [CHAT_PROMPT]
    
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
        model="gemma2-9b-it",
        messages=chat_messages,
        max_tokens=100,
        temperature=1.2
    )
    
    return response.choices[0].message.content

# Create Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="ISP 🐬 AI Club 🤖 Project #2: Creating a Safe Chatbot",
    description="Chat with a Groq-powered LLM assistant (with safety filtering)",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)
