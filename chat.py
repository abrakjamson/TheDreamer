import os
import gradio as gr
from gradio import ChatMessage
import asyncio
from autogen_core.components.models import SystemMessage, UserMessage

from wetware import ModelClient, LLMResult
from the_dreamer import begin_dreaming

# Initialize the model client
model_client = ModelClient()

chat_system_prompt = "You are a helpful assistant."

async def generate_response(message, history):
    messages = [ChatMessage(role="system", content=chat_system_prompt)]
    
    # Add chat history
    for h in history:
        messages.append(ChatMessage(role=h.role, content=h.content))
    
    # Add current message
    messages.append(ChatMessage(role="user", content=message))
    
    response = await model_client.create(messages, cancellation_token=None)
    assistant_response = response.content
    
    # Append the assistant's response to the history
    history.append(ChatMessage(role="user", content=message))
    history.append(ChatMessage(role="assistant", content=assistant_response))
    
    return ChatMessage(role="assistant", content=assistant_response)

# Load content from text files for the right column
def load_text_file(filename):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return file.read()
    else:
        return f"{filename} not found."

# Gradio interface components
with gr.Blocks() as demo:
    gr.Markdown("# Chat with The Dreamer")

    with gr.Row():
        chat_interface = gr.ChatInterface(
            fn=generate_response,
            chatbot=gr.Chatbot(),
            type='messages'
        )

        # Display the text files on the right pane
        with gr.Column():
            with gr.Accordion("Image of Self"):
                gr.TextArea(value=lambda: load_text_file("image_of_self.txt"), interactive=False)
            
            with gr.Accordion("Goals"):
                gr.TextArea(value=lambda: load_text_file("goals.txt"), interactive=False)

async def start_background_tasks():
    await begin_dreaming()

if __name__ == "__main__":
    # Start the event loop and run the background tasks
    loop = asyncio.get_event_loop()
    loop.create_task(start_background_tasks())

    # Start Gradio
    demo.launch()