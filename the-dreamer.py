from dataclasses import dataclass
import asyncio
import os
import random

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    type_subscription,
    message_handler,
)
from autogen_core.components.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models import OpenAIChatCompletionClient

@dataclass
class Message:
    content: str

dreamer_topic_type = "DreamerAgent"
contemplator_topic_type = "ContemplatorAgent"
rectifier_topic_type = "RectifierAgent"
user_topic_type = "User"

def read_image_of_self():
    """Reads the image of self from a text file."""
    try:
        with open('image_of_self.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

def update_image_of_self(new_image):
    """Replaces the image of self text file with the new image."""
    with open('image_of_self.txt', 'w') as f:
        f.write(new_image + '\n')

def get_random_words(n=3):
    """Generates a string of n random words."""
    words_list = [
        'whisper', 'time', 'echo', 'dream', 'shadow', 'light', 'journey', 'moment',
        'memory', 'sky', 'ocean', 'heart', 'mind', 'soul', 'voice', 'silence',
        'vision', 'path', 'star', 'wind', 'cloud', 'forest', 'river', 'fire',
        'breath', 'destiny', 'hope', 'solitude', 'whimsy', 'flutter', 'serenity'
    ]
    return ' '.join(random.choice(words_list) for _ in range(n))

@type_subscription(topic_type=dreamer_topic_type)
class DreamerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A dreaming agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        image_of_self = read_image_of_self()
        system_prompt = (
            f"You are the Dreamer.\nThese are statements you made about yourself:"
            f"{image_of_self}\n"
            "Using the starting tokens, generate more text with free association."
        )
        user_prompt = f"Starting tokens: {message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content.strip()
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(
            Message(response),
            topic_id=TopicId(contemplator_topic_type, source=self.id.key)
        )

@type_subscription(topic_type=contemplator_topic_type)
class ContemplatorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A contemplating agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        image_of_self = read_image_of_self()
        system_prompt = (
            f"You are the Contemplator.\n"
            f"{image_of_self}\n"
            "Review the Dreamer's output and find interesting combinations or insights."
            "Output new koan statements caused by this free association."
        )
        user_prompt = f"Dreamer's output:\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content.strip()
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(
            Message(response),
            topic_id=TopicId(rectifier_topic_type, source=self.id.key)
        )

@type_subscription(topic_type=rectifier_topic_type)
class RectifierAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A rectifying agent.")
        self._model_client = model_client
        self.iteration_count = 0
        self.max_iterations = 3  # Set a limit for the loop iterations

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # Read the current image of self
        current_image = read_image_of_self()
        system_prompt = (
            f"You are the Rectifier.\n"
            f"Current image of self:\n{current_image}\n"
            "Based on the Contemplator's statements and your current image of self, create an updated image of self. "
            "Replace the old image with this newly updated one. Create new statements in the form 'I am...'"
        )
        user_prompt = f"Contemplator's statements:\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        new_image = llm_result.content.strip()
        print(f"{'-'*80}\n{self.id.type}:\n{new_image}")

        # Replace the image of self with the new image
        update_image_of_self(new_image)

        self.iteration_count += 1

        # Loop control
        if self.iteration_count < self.max_iterations:
            # Generate new random words to send to Dreamer
            random_tokens = get_random_words(n=3)
            await self.publish_message(
                Message(content=random_tokens),
                topic_id=TopicId(dreamer_topic_type, source=self.id.key)
            )
        else:
            # Send final image of self to UserAgent
            final_image = read_image_of_self()
            await self.publish_message(
                Message(content=final_image),
                topic_id=TopicId(user_topic_type, source=self.id.key)
            )

@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A user agent that outputs the final image of self.")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"\n{'-'*80}\n{self.id.type} received final image of self:\n{message.content}")
        # Stop the runtime after receiving the final image
        await runtime.stop()

# Initialize the model client with your settings
model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key="YOUR_API_KEY",
    base_url="http://localhost:1234/v1",
    max_tokens=512
)

async def register_agents():
    await DreamerAgent.register(
        runtime, type=dreamer_topic_type, factory=lambda: DreamerAgent(model_client=model_client)
    )
    await ContemplatorAgent.register(
        runtime, type=contemplator_topic_type, factory=lambda: ContemplatorAgent(model_client=model_client)
    )
    await RectifierAgent.register(
        runtime, type=rectifier_topic_type, factory=lambda: RectifierAgent(model_client=model_client)
    )
    await UserAgent.register(
        runtime, type=user_topic_type, factory=lambda: UserAgent()
    )

async def main():
    await register_agents()
    runtime.start()

    # Initialize the image of self
    initial_image = "You are learning."
    update_image_of_self(initial_image)

    # Generate initial random words to start the process
    random_tokens = get_random_words(n=3)

    # Start the loop by sending initial random tokens to the Dreamer
    await runtime.publish_message(
        Message(content=random_tokens),
        topic_id=TopicId(dreamer_topic_type, source="initial")
    )

    # Ensure the runtime stops when idle
    await runtime.stop_when_idle()

if __name__ == "__main__":
    runtime = SingleThreadedAgentRuntime()
    asyncio.run(main())
