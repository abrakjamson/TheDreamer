"""
Copyright Abram Jackson 2024
ALl Rights Reserved
"""

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

starting_sense_of_self = "You are learning."
dreamer_prompt_part1 = """You are the Dreamer.
These are statements you made about yourself:
"""
dreamer_prompt_part2 = \
"""\nUsing the starting tokens, generate more text with free association."""
contemplator_prompt_part1 = \
"""You are the Contemplator.\n"""
contemplator_prompt_part2 = \
"""\nReview the Dreamer's output and find interesting combinations or insights.
Output new koan statements caused by this free association."""
rectifier_prompt_part1 = \
"""You are the Rectifier.
Current image of self:\n"""
rectifier_prompt_part2 = \
"""\nBased on the Contemplator's statements and your current image of self, create an updated image of self.
"Replace the old image with this newly updated one. Create new statements in the form 'I am...'"""

# Perform this many loops before stopping
# TODO allow for infinite loops
MAX_ITERATIONS = 3

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
    # I used diceware.dmuth.org to pick these words
    words_list = [
        'zodiac', 'clapping', 'stumbling' 'truce', 'smilingly', 'waving', 'mourner',
        'scrutiny', 'walkt', 'reimburse', 'skimming', 'atrium', 'refreeze', 'entrust',
        'cobweb', 'judgingly', 'plunging', 'patience', 'disarray', 'spearhead', 'aging',
        'uncounted', 'timing', 'reanalyze', 'scrabble', 'reshoot', 'pagan', 'quintuple',
        'landmine', 'reconvene', 'prong', 'strict', 'boneless', 'dazzler', 'tinwork'
    ]
    return ' '.join(random.choice(words_list) for _ in range(n))

@type_subscription(topic_type=dreamer_topic_type)
class DreamerAgent(RoutedAgent):
    """
    The Dreamer agent takes sensory input of random words and crafts narrative around them.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A dreaming agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        image_of_self = read_image_of_self()
        system_prompt = dreamer_prompt_part1 + image_of_self + dreamer_prompt_part2
        
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
    """
    The Contemplator thinks about the Dreamer's dreams to devise lessons and potential learnings.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A contemplating agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        image_of_self = read_image_of_self()
        system_prompt = contemplator_prompt_part1 + image_of_self + contemplator_prompt_part2

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
    """
    The Rectifier determines what learnings are compatible with its existing sense of self.
    It updates how it thinks about itself based on what it has learned.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A rectifying agent.")
        self._model_client = model_client
        self.iteration_count = 0
        self.max_iterations = MAX_ITERATIONS  # Set a limit for the loop iterations

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # Read the current image of self
        current_image = read_image_of_self()
        system_prompt = rectifier_prompt_part1 + current_image + rectifier_prompt_part2
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
            print(f"New sensory input: {random_tokens}")
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
    """
    The User is only used for outputting the final results at this time.
    """
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
    initial_image = starting_sense_of_self
    update_image_of_self(initial_image)

    # Generate initial random words to start the process
    random_tokens = get_random_words(n=3)
    print(f"Sensory input: {random_tokens}")

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
