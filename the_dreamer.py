"""
Copyright Abram Jackson 2024
ALl Rights Reserved
"""

# Import the ModelClient from wetware

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
from wetware import ModelClient

@dataclass
class Message:
    content: str

dreamer_topic_type = "DreamerAgent"
contemplator_topic_type = "ContemplatorAgent"
rectifier_topic_type = "RectifierAgent"
commander_topic_type = "CommanderAgent"

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
Your goal is:\n"""
rectifier_prompt_part2 = \
"""Current image of self:\n"""
rectifier_prompt_part3 = \
"""\nBased on your goal, the Contemplator's statements, and your current image of self, create an updated image of self.
Take what you've learned and apply them to yourself to grow and change, better able to accomplish your goal.
"Replace the old image with this newly updated one. Create new statements in the form 'I am...'"""
commander_prompt_part1 = \
"""You are the Commander. You choose a goal to achieve by your thoughts based on your image of self and purpose.
Current image of self:\n"""
commander_prompt_part2 = \
"""Replace the current goal with an updated goal if you wish. It should be tangible and actionable. Write a short statement of your new goal, or the same goal. No yapping!"""

# Perform this many loops before updating the goal
iterations_for_goal = 5

# Update the goal this many times before stopping
# Total iterations will be iterations_for_goal * goal_iterations
# the number of LLM calls is (iterations_for_goal * 3 * goal_iterations) + goal_iterations
goal_iterations = 2

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

def read_thoughts():
    """Reads the thoughts from a text file."""
    try:
        with open('thoughts.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

def update_thoughts(new_thoughts):
    """Replaces the thoughts text file with the new thoughts."""
    with open('thoughts.txt', 'w') as f:
        f.write(new_thoughts + '\n')

def read_goals():
    """Reads the goals from a text file."""
    try:
        with open('goals.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

def update_goals(new_goals):
    """Replaces the goals text file with the new goals."""
    with open('goals.txt', 'w') as f:
        f.write(new_goals + '\n')

def get_random_words(n=3):
    """Generates a string of n random words."""
    # I used diceware.dmuth.org to pick these words
    words_list = [
        'zodiac', 'clapping', 'stumbling', 'truce', 'smilingly', 'waving', 'mourner',
        'scrutiny', 'walk', 'reimburse', 'skimming', 'atrium', 'refreeze', 'entrust',
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
        update_thoughts(f"THE DREAMER\n{response}")
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
        update_thoughts(f"THE CONTEMPLATOR\n{response}")
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
        self.max_iterations = iterations_for_goal  # Set a limit for the loop iterations

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # Read the current image of self
        current_image = read_image_of_self()
        system_prompt = rectifier_prompt_part1 + current_goal + rectifier_prompt_part2 + current_image + rectifier_prompt_part3
        user_prompt = f"Contemplator's statements:\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        new_image = llm_result.content.strip()
        update_thoughts(f"THE RECTIFIER\n{new_image}")
        print(f"{'-'*80}\n{self.id.type}:\n{new_image}")

        # Replace the image of self with the new image
        update_image_of_self(new_image)

        self.iteration_count += 1
        print(f"Iteration count: {self.iteration_count}")

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
            # Determine whether to update the goal
            await self.publish_message(
                Message(content=""), # The Commander only uses the image of self text as input
                topic_id=TopicId(commander_topic_type, source=self.id.key)
            )

@type_subscription(topic_type=commander_topic_type)
class CommanderAgent(RoutedAgent):
    """
    The Commander Agent sets a goal every few turns based on its understanding of self.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A agent that sets a goal for all the agents")
        self._model_client = model_client
        self.goal_count = 0
        self.max_iterations = goal_iterations

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        current_image = read_image_of_self()
        system_prompt = commander_prompt_part1 + current_image + commander_prompt_part2
        
        user_prompt = read_goals()
        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        current_goal = llm_result.content.strip()
        update_thoughts(f"THE COMMANDER\n{current_goal}")
        update_goals(current_goal)
        print(f"\n{'-'*80}\n{self.id.type} has set a goal:\n{current_goal}")

        self.goal_count += 1
        # TODO allow for continuous execution
        if self.goal_count < self.max_iterations:
            # Loop again back to the Dreamer.
            self.iteration_count = 0
            random_tokens = get_random_words(n=3)
            print(f"New sensory input: {random_tokens}")
            await self.publish_message(
                Message(content=random_tokens),
                topic_id=TopicId(dreamer_topic_type, source=self.id.key)
            )
        else:
            await runtime.stop()

# Initialize the model client using the ModelClient from wetware.py
model_client = ModelClient()

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
    await CommanderAgent.register(
        runtime, type=commander_topic_type, factory=lambda: CommanderAgent(model_client=model_client)
    )

async def begin_dreaming():
    runtime = SingleThreadedAgentRuntime()
    await register_agents()
    runtime.start()

    # Initialize the image of self
    update_image_of_self(starting_sense_of_self)

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
    asyncio.run(begin_dreaming())
