"""
Copyright Abram Jackson 2024
ALl Rights Reserved
"""

# Import the ModelClient from wetware

from dataclasses import dataclass
import asyncio
import os
import random
import re

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    type_subscription,
    message_handler,
)
from autogen_core.components.models import ChatCompletionClient, SystemMessage, UserMessage, AssistantMessage
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
dreamer_prompt_part1 = """You are the Dreamer, one part of a mind.
These are statements you made about yourself:
"""
dreamer_prompt_part2 = \
"""\nUsing the starting tokens, generate more text with free association."""
contemplator_prompt_part1 = \
"""You are the Contemplator, one part of a mind.\n"""
contemplator_prompt_part2 = \
"""\nReview the Dreamer's output and find interesting combinations or insights.
Think about them, and then output new koan statements caused by this free association into a fenced Markdown code block."""
rectifier_prompt_part1 = \
"""You are the Rectifier, one part of a mind.
Your goal is:\n"""
rectifier_prompt_part2 = \
"""Current image of self:\n"""
rectifier_prompt_part3 = \
"""\nBased on your goal, the Contemplator's statements, and your current image of self, create an updated image of self.
Take what you've learned and apply them to yourself to grow and change, better able to accomplish your goal.
"Replace the old image with this newly updated one. Create new statements in the form 'I am...' into a fenced Markdown code block"""
commander_prompt_part1 = \
"""You are the Commander, part of a conscious mind.
You choose a goal to achieve by your thoughts based on your image of self and purpose.
Current image of self:\n"""
commander_prompt_part2_has_goal = \
"""\nThe goal you previously set for yourself:\n"""
commander_prompt_part2_no_goal = \
"""\nYou have not yet set a goal for yourself. Review your image of self and set a goal to achieve.
* It should be tangible and actionable.
* It should begin, "I will"
"""
commander_prompt_part3_has_goal = \
"""\nReplace the current goal with an updated goal if you wish. It should be tangible and actionable.
Write a short statement of your new goal, or the same goal.
* It should be tangible and actionable.
* It should begin, "I will"
"""

# Perform this many loops before updating the goal
iterations_for_goal = 3

# Update the goal this many times before stopping
# Total iterations will be iterations_for_goal * goal_iterations
# the number of LLM calls is (iterations_for_goal * 3 * goal_iterations) + goal_iterations
goal_iterations = 5

#########################
# Helper functions
#########################
def update_dreams(new_dream):
    """Replaces the image of self text file with the new image."""
    with open('dreams.txt', 'w') as f:
        f.write(new_dream)

def update_musings(new_musing):
    """Replaces the image of self text file with the new image."""
    with open('musings.txt', 'w') as f:
        f.write(new_musing)

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
        f.write(new_image)

def read_musings():
    """Reads the thoughts from a text file."""
    try:
        with open('musings.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

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
        f.write(new_goals)

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

def extract_thought(markdown_string):
    # Agents are intsructed to write their thoughts to communicate in a Markdown code block
    code_block_pattern = r"```(?:[^`\n]*\n)?(.*?)```"
    code_blocks = re.findall(code_block_pattern, markdown_string, re.DOTALL)
    
    # Return the content of the first code block, or an empty string if no code block is found
    return code_blocks[0].strip() if code_blocks else ""

########################
# Agents
########################
@type_subscription(topic_type=dreamer_topic_type)
class DreamerAgent(RoutedAgent):
    """
    The Dreamer agent takes sensory input of random words and crafts narrative around them.
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A dreaming agent.")
        self._model_client = model_client
        self.firstTurn = True

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if self.firstTurn:
            # On the first turn, the Dreamer is entirely bootstrapped from input; no programming
            self.firstTurn = False
            assistant_preprompt = f"I dream of {message.content}"
            llm_result = await self._model_client.create(
                messages=[
                    AssistantMessage(content=assistant_preprompt, 
                                     type="AssistantMessage",
                                     source=self.id.key)
                ],
                cancellation_token=ctx.cancellation_token,
            )
        else:
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
        update_dreams(response)
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
        response = extract_thought(llm_result.content)
        update_musings(llm_result.content)
        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}")

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
        current_goal = read_goals()
        system_prompt = rectifier_prompt_part1 + current_goal + rectifier_prompt_part2 + current_image + rectifier_prompt_part3
        user_prompt = f"Contemplator's statements:\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        new_image = extract_thought(llm_result.content)
        if new_image == "":  # If the Rectifier has no new image of self, keep the old one
            new_image = current_image
        update_image_of_self(new_image)
        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}")

        # every few turns, update the goal
        self.iteration_count += 1
        print(f"Iteration count: {self.iteration_count}")
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
        current_goal = read_goals()
        if current_goal == "":
            system_prompt = commander_prompt_part1 + current_image + commander_prompt_part2_no_goal
        else:
            system_prompt = commander_prompt_part1 + current_image + commander_prompt_part2_has_goal + current_goal + commander_prompt_part3_has_goal
        
        
        previous_goal = read_goals()
        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=previous_goal, source=self.id.key)
            ],
            cancellation_token=ctx.cancellation_token,
        )
        current_goal = llm_result.content.strip()
        update_goals(current_goal)
        print(f"\n{'-'*80}\n{self.id.type} has thought:\n{llm_result.content}")

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

########################
# Initialization
########################

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
    """Begin. Called by either the __main__ block or the Gradio interface."""
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

runtime = SingleThreadedAgentRuntime()

if __name__ == "__main__":
    asyncio.run(begin_dreaming())