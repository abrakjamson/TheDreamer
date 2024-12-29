import os
import gradio as gr
from gradio import ChatMessage
import asyncio
from autogen_core.components.models import SystemMessage, UserMessage

from wetware import ModelClient, LLMResult
from the_dreamer import begin_dreaming

# Initialize the model client
model_client = ModelClient()

def read_image_of_self():
    """Reads the image of self from a text file."""
    try:
        with open('image_of_self.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

async def generate_response(message, history):
    messages = [ChatMessage(role="system", content=read_image_of_self())]
    
    # Add chat history
    for h in history:
        messages.append(ChatMessage(role=h['role'], content=h['content']))
    
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

# Function to reload all text files
def reload_text_files():
    return (
        load_text_file("image_of_self.txt"),
        load_text_file("goals.txt"),
        load_text_file("thoughts.txt")
    )

# Gradio interface components
with gr.Blocks() as demo:
    gr.Markdown("# Chat with The Dreamer")

    with gr.Column():
        chat_interface = gr.ChatInterface(
            fn=generate_response,
            chatbot=gr.Chatbot(type="messages"),
            type="messages"
        )

    # Display the text files on the right pane
    with gr.Column():
        with gr.Accordion("Image of Self"):
            image_of_self = gr.TextArea(value=lambda: load_text_file("image_of_self.txt"), interactive=False)
        
        with gr.Accordion("Goals"):
            goals = gr.TextArea(value=lambda: load_text_file("goals.txt"), interactive=False)

        with gr.Accordion("Thoughts"):
            thoughts = gr.TextArea(value=lambda: load_text_file("thoughts.txt"), interactive=False)
        
        # Add refresh button
        refresh_btn = gr.Button("Refresh Content")
        refresh_btn.click(
            fn=reload_text_files,
            outputs=[
                image_of_self,
                goals,
                thoughts
            ]
        )

# Apparently, all of the following are required to run two things at once in Python
async def start_background_tasks():
    print("Starting dreaming process...")
    await begin_dreaming()

async def start_demo():
    # Launch demo without await since it returns TupleNoPrint
    demo.queue().launch(prevent_thread_lock=True, show_error=True)
    # Keep the coroutine running
    while True:
        await asyncio.sleep(1)

async def main():
    print("Starting background tasks and demo...")
    try:
        await asyncio.gather(
            start_background_tasks(),
            start_demo()
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())