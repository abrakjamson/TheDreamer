import os
import gradio as gr
from gradio import ChatMessage
import asyncio
from autogen_core.components.models import SystemMessage, UserMessage

from wetware import ModelClient, LLMResult
from the_dreamer import begin_dreaming

# Initialize the model client
model_client = ModelClient()

chat_prompt_part1 = \
"""You are The Dreamer. You are chatting with a human, which is the only means you have to achieve your goals.
The goals you have set for yourself are:"""
chat_prompt_part2 = \
"""You have written the following about yourself:"""

def read_image_of_self():
    """Reads the image of self from a text file."""
    try:
        with open('image_of_self.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

async def generate_response(message, history):
    prompt = chat_prompt_part1 + "\n" + load_text_file("goals.txt") + "\n" + chat_prompt_part2 + "\n" + load_text_file("image_of_self.txt")
    messages = [ChatMessage(role="system", content=prompt)]
    
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
        return f"{filename} not yet created."

# Function to reload all text files
def reload_text_files():
    return (
        load_text_file("dreams.txt"),
        load_text_file("musings.txt"),
        load_text_file("image_of_self.txt"),
        load_text_file("goals.txt"),
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
        with gr.Accordion("Most recent thoughts"):
            dreams = gr.TextArea(value=lambda: load_text_file("dreams.txt"),
                                    interactive=False,
                                    label="Dreams from the Dreamer",
                                    every=5)
            musings = gr.TextArea(value=lambda: load_text_file("musings.txt"), 
                                   interactive=False,
                                   label="Musings from the Contemplator",
                                   every=5)
            image_of_self = gr.TextArea(value=lambda: load_text_file("image_of_self.txt"), 
                                        interactive=False,
                                        label="Image of Self from the Rectifier",
                                        every=5)
            goals = gr.TextArea(value=lambda: load_text_file("goals.txt"),
                                interactive=False,
                                label="Goals from the Commander",
                                every=5)       

        refresh_btn = gr.Button("Refresh Content")
        refresh_btn.click(
            fn=reload_text_files,
            outputs=[
                dreams,
                musings,
                image_of_self,
                goals
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