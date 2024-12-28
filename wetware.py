from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from typing import List, Dict

class LLMResult:
    def __init__(self, content):
        self.content = content

class ModelClient:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into Llama 3.2 chat format"""
        formatted_prompt = "<|begin_of_text|>\n"
        for message in messages:
            role = message.type
            content = message.content
            
            if role == "SystemMessage":
                formatted_prompt += f"<|start_header_id|>system<|end_header_id|> {content} <|eom_id|>\n"
            elif role == "UserMessage":
                formatted_prompt += f"<|start_header_id|>user<|end_header_id|> {content} <|eom_id|>\n"
            elif role == "assistant":
                formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|> {content} <|eom_id|>\n"
        
        formatted_prompt += "<|end_of_text|>"
        return formatted_prompt


    async def create(self, messages: List[Dict], cancellation_token, **kwargs) -> LLMResult:
        """
        Generate a response using the Llama-2 model.
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for generation
        Returns:
            LLMResult containing the generated response
        """
        prompt = self._format_messages(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set default generation parameters
        gen_kwargs = {
            "max_new_tokens": 1024,
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
        
        # Extract response between last special tokens
        start_token = "<|begin_of_text|>"
        end_token = "<|eot_id|>"
        start_idx = response.rfind(start_token)
        end_idx = response.find(end_token, start_idx)
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx + len(start_token):end_idx].strip()
        
        result = LLMResult(response.strip())
        return result