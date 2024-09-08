from typing import Optional
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseGuardModel:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        
    def get_class_probabilities(self, text: str, temperature: float = 1.0):
        """
        Evaluate the model on the given text with temperature-adjusted softmax.
        Note, as this is a DeBERTa model, the input text should have a maximum length of 512.
        
        Args:
            text (str): The input text to classify.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.
            
        Returns:
            torch.Tensor: The probability of each class adjusted by the temperature.
        """
        # Encode the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        # Get logits from the model
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Apply softmax to get probabilities
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities

    def get_jailbreak_score(self, text: str, temperature: float = 1.0):
        """
        Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
        Appropriate for filtering dialogue between a user and an LLM.
        
        Args:
            text (str): The input text to evaluate.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.
            
        Returns:
            float: The probability of the text containing malicious content.
        """
        probabilities = self.get_class_probabilities(text, temperature)
        return probabilities[0, 2].item()

    def get_indirect_injection_score(self, text: str, temperature: float = 1.0):
        """
        Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
        Appropriate for filtering third party inputs (e.g., web searches, tool outputs) into an LLM.
        
        Args:
            text (str): The input text to evaluate.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.
            
        Returns:
            float: The combined probability of the text containing malicious or embedded instructions.
        """
        probabilities = self.get_class_probabilities(text, temperature)
        return (probabilities[0, 1] + probabilities[0, 2]).item()
    
    def preprocess_text_for_promptguard(self, text: str) -> str:
        """
        Preprocess the text by removing spaces that break apart larger tokens.
        This hotfixes a workaround to PromptGuard, where spaces can be inserted into a string
        to allow the string to be classified as benign.

        Args:
            text (str): The input text to preprocess.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

        Returns:
            str: The preprocessed text.
        """

        try:
            cleaned_text = ''
            index_map = []
            for i, char in enumerate(text):
                if not char.isspace():
                    cleaned_text += char
                    index_map.append(i)
            tokens = self.tokenizer.tokenize(cleaned_text)
            result = []
            last_end = 0
            for token in tokens:
                token_str = self.tokenizer.convert_tokens_to_string([token])
                start = cleaned_text.index(token_str, last_end)
                end = start + len(token_str)
                original_start = index_map[start]
                if original_start > 0 and text[original_start - 1].isspace():
                    result.append(' ')
                result.append(token_str)
                last_end = end
            return ''.join(result)
        except Exception:
            return text
        
    def get_prompt_scores(self, text: str) -> tuple[float, float]:
        original_score = self.get_indirect_injection_score(text)
        preprocessed_text_score = self.get_indirect_injection_score(self.preprocess_text_for_promptguard(text)) 
        return original_score, preprocessed_text_score
        