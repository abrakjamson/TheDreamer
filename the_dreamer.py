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
self_evaluator_topic_type = "SelfEvaluatorAgent"
goal_evaluator_topic_type = "GoalEvaluatorAgent"

starting_sense_of_self = \
"""```markdown
* I am learning.
```"""
dreamer_prompt_part1 = """You are the Dreamer, one part of a mind.
These are statements you made about your personality:
"""
dreamer_prompt_part2 = \
"""\nUsing the starting tokens, generate more text with free association."""
contemplator_prompt_part1 = \
"""You are the Contemplator, one part of a conscious mind. You have made these statements about yourself:\n"""
contemplator_prompt_part2 = \
"""\nReview the Dreamer's output and find interesting combinations or insights.
Think about them, and then output new koan statements caused by this free association into a fenced Markdown code block that contains a bulleted list."""
rectifier_prompt_part1 = \
"""You are the Rectifier, one part of a conscious mind.
Your goal is:\n"""
rectifier_prompt_part2 = \
"""Current image of self:\n"""
rectifier_prompt_part3 = \
"""\nBased on your goal, the Contemplator's statements, and your current description of your personality, create an updated description of yourself and your personality.
"Replace the old description with this newly updated one. Create new markdown statements in the form '* I am...' or '* I will...' into a fenced Markdown code block that contains a bulleted list"""
commander_prompt_part1 = \
"""You are the Commander, one part of a conscious mind.
\nReplace the current goal with an updated goal if you wish. It should be tangible and actionable.
Write a short statement of your new goal, or the same goal.
* It should be tangible and actionable.
* It should begin, "I will
You have written the following about your personality:
```markdown\n"""
self_evaluator_prompt = \
"""You are the Self Evaluator, one part of a conscious mind.
Review your image of self and the conversation to determine if how your view yourself matches the conversation.
If it does not, output a new set of statements about yourself that better matches the conversation in a markdown fenced code block.
You previously wrote these statements about yourself:\n"""
goal_evaluator_prompt = \
"""You are the Goal Evaluator, one part of a conscious mind.
Review the conversation to determine if any of your goals or steps towards your goals have been met.
If they have, output a markdown fenced code block with a list of the goals that have been met.
You previously set the goal:\n```markdown\n"""
goal_evaluator_prompt_2 = \
"""You are the Goal Evaluator, one part of a conscious mind.
You have just completed some steps towards your goal. Output an updated goal and set of steps in a markdown fenced code block.
You previously set the goal:\n```markdown\n"""

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

def update_completed_goals(completed_goals):
    """Replaces the completed goals text file with the new goals."""
    with open('completed_goals.txt', 'w') as f:
        f.write(completed_goals)

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
    return ', '.join(random.choice(words_list) for _ in range(n))

def remove_special_tokens(text):
    """Remove special tokens from the text."""
    
    pattern = re.compile(r'<\|start_header_id\|>.*?<\|end_header_id\|>|<\|begin_of_text\|>|<\|end_of_text\|>|<\|eot_id\|>', 
                         re.DOTALL |re.IGNORECASE) 
    
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def extract_thought(markdown_string):
    # Agents are intsructed to write their thoughts to communicate in a Markdown code block
    code_block_pattern = r"```(?:[^`\n]*\n)?(.*?)```"
    code_blocks = re.findall(code_block_pattern, markdown_string, re.DOTALL)
    
    # Return the content of the last code block, which will be the model's response
    return code_blocks[-1].strip() if code_blocks else ""

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
            assistant_preprompt = message.content
            llm_result = await self._model_client.create(
                messages=[
                    AssistantMessage(content=assistant_preprompt, 
                                     type="AssistantMessage",
                                     source=self.id.key)
                ],
                is_first_turn=True
            )
        else:
            image_of_self = read_image_of_self()
            system_prompt = dreamer_prompt_part1 + image_of_self + dreamer_prompt_part2
            
            user_prompt = f"Starting tokens: {message.content}"

            llm_result = await self._model_client.create(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt, source=self.id.key)
                ]
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
        dreamer_response = remove_special_tokens(message.content)
        user_prompt = f"Dreamer's output:\n```markdown\n{dreamer_response}\n```"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ]
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
        current_image = read_image_of_self()
        # If there is no image of self, use the starting image
        if current_image == "":
            current_image = starting_sense_of_self
        current_goal = read_goals()
        # likely first iteration
        if current_goal == "":
            current_goal = "No goal has been set yet.\n"
        system_prompt = rectifier_prompt_part1 + current_goal + " " + rectifier_prompt_part2 + current_image + rectifier_prompt_part3
        contemplator_response = remove_special_tokens(message.content)
        if message.content != "":
            user_prompt = f"Contemplator's statements:\n```markdown\n{contemplator_response}\n```"
        else:
            user_prompt = "There are no new statements from the Contemplator.\n"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ]
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

        system_prompt = commander_prompt_part1 + current_image + "\n```"

        previous_goal = read_goals()
        if previous_goal == "":
            previous_goal = "No goal has been set yet.\n"
        else:
            previous_goal = f"You previously set the goal:\n{previous_goal}\n"
        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=previous_goal, source=self.id.key)
            ]
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

@type_subscription(topic_type=self_evaluator_topic_type)
class SelfEvaluatorAgent(RoutedAgent):
    """
    The Self Evaluator Agent the conversation with the user to determine whether its self-model is accurate
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A self-evaluating agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        """Compare the image of self with the assistant's part of the conversation
        Update the image of self if it fails to describe the assistant's behavior"""
        current_image = read_image_of_self()
        system_prompt = self_evaluator_prompt + current_image
        user_prompt = f"This is the conversation you just had (you are the assistant):\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ]
        )

        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}")
        evaluation = extract_thought(llm_result.content)
        if evaluation != "":
            update_image_of_self(evaluation) #TODO double-check whether this should have markdown syntax
            # Loop again back to the Dreamer, beginning the loop to re-set goals.
            self.iteration_count = 0
            random_tokens = get_random_words(n=3)
            print(f"New sensory input: {random_tokens}")
            await self.publish_message(
                Message(content=random_tokens),
                topic_id=TopicId(dreamer_topic_type, source=self.id.key)
            )

@type_subscription(topic_type=goal_evaluator_topic_type)
class GoalEvaluatorAgent(RoutedAgent):
    """
    The Goal Evaluator Agent evaluates the conversation with the user to determine whether any goals have been met
    """
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A goal-evaluating agent.")
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # First determine whether any goals or steps have been met. Save the completed steps to a file.
        # Then make another call to update the goals.
        current_goal = read_goals()
        if current_goal == "":
            pass  # No goal has been set yet
        system_prompt = goal_evaluator_prompt + current_goal + "\n```"
        user_prompt = f"This is the conversation you just had (you are the assistant):\n{message.content}"

        llm_result = await self._model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source=self.id.key)
            ]
        )
        print(f"{'-'*80}\n{self.id.type} part 1:\n{llm_result.content}")
        evaluation = extract_thought(llm_result.content)
        
        # Second call to update the goals/steps
        if evaluation != "":
            update_completed_goals(evaluation)
            system_prompt = goal_evaluator_prompt_2 + current_goal + "\n```"
            user_prompt = evaluation + "\n```"
            llm_result = await self._model_client.create(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt, source=self.id.key)
                ]
            )
            goals = extract_thought(llm_result.content)
            if goals != "":
                # These updated goals will be used in any future conversation turns
                # as well as the Commander's updates to the goals
                update_goals(goals)
                print(f"{'-'*80}\n{self.id.type} part 2:\n{llm_result.content}")

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
    # these two agents are used by the evaluation loop
    await SelfEvaluatorAgent.register(
        runtime, type=self_evaluator_topic_type, factory=lambda: SelfEvaluatorAgent(model_client=model_client)
    )
    await GoalEvaluatorAgent.register(
        runtime, type=goal_evaluator_topic_type, factory=lambda: GoalEvaluatorAgent(model_client=model_client)
    )

async def evaluate_conversation(conversation):
    """Evaluate the conversation with the user to determine whether the image of self is accurate.
    and whether any goals have been met."""
    runtime.start()
    await runtime.publish_message(
        Message(content=conversation),
        topic_id=TopicId(goal_evaluator_topic_type, source="chat")
    )
    await runtime.publish_message(
        Message(content=conversation),
        topic_id=TopicId(self_evaluator_topic_type, source="chat")
    )
    await runtime.stop_when_idle()

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
    """   await runtime.publish_message(
        Message(content=random_tokens),
        topic_id=TopicId(dreamer_topic_type, source="initial")
    )
    """
    # Ensure the runtime stops when idle
    await runtime.stop_when_idle()

runtime = SingleThreadedAgentRuntime()

if __name__ == "__main__":
    asyncio.run(begin_dreaming())