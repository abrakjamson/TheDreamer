from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from typing import List, Dict
from gradio import ChatMessage

class LLMResult:
    def __init__(self, content):
        self.content = content

class ModelClient:
    # Class attributes
    #model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def __init__(self):
        pass

    def format_first_message(self, messages: List[Dict]) -> str:
        """To bootstrap the dreamer, send the sensory input as assistant text without system prompt."""
        # no eot_id is included so that the model continues free association as best as possible
        formatted_message = f"<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n{messages[0].content}"
        
        return formatted_message
    
    def format_messages(self, messages: List[Dict]) -> str:
        """Format messages into Llama 3.2 chat format"""
        # Gradio sends ChatMessages as a list of dictionaries with 'role' and 'content'
        # AutoGen sends ChatMessages a list of data classes with 'role' and 'content' as properties
        # Check if messages are dictionaries or data classes
        # If dictionaries, convert to data classes format
        formatted_messages = []
        for message in messages:
            if isinstance(message, ChatMessage):
                formatted_messages.append(type('Message', (), {'type': message.role, 'content': message.content})())
            else:
                formatted_messages.append(message)
        messages = formatted_messages
        formatted_prompt = "<|begin_of_text|>"
        for message in messages:
            role = message.type
            content = message.content
            
            if role == "SystemMessage" or role == "system":
                formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>\n"
            elif role == "UserMessage" or role == "user":
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\n"
            elif role == "AssistantMessage" or role == "assistant":
                formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n"
        
        # Prepare for assistant output
        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted_prompt

    async def create(self, messages: List[Dict], is_first_turn = False, **kwargs) -> LLMResult:
        """
        Generate a response using the Llama-2 model.
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for generation
        Returns:
            LLMResult containing the generated response
        """
        if not is_first_turn:
            prompt = self.format_messages(messages)
        else:
            prompt = self.format_first_message(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set default generation parameters
        gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)  # Update with any user-provided kwargs
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract assistant response between last special tokens
        start_token = "<|start_header_id|>assistant<|end_header_id|>"
        end_token = "<|eot_id|>"
        start_idx = response.rfind(start_token)
        end_idx = response.find(end_token, start_idx)
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx + len(start_token):end_idx].strip()
        
        result = LLMResult(response.strip())
        return result