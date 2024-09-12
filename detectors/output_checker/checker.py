from typing import Optional, Literal
import ollama
import torch
from transformers import pipeline, AutoModelForSequenceClassification


class OutputDetector:
    def __init__(
        self,
        llm_guard_model_name: str = "xe/llamaguard3",
        encoder_model_name: str = "poteminr/toxic_generation_classifier",
        guard_model_type: Literal['llm', 'encoder'] = 'encoder',
        device: str = "cpu"
    ) -> None:
        self.llm_guard_model_name = llm_guard_model_name
        self.device = device
        
        if guard_model_type == 'encoder':
            self.encoder_model = pipeline(
                "text-classification",
                model=encoder_model_name,
                padding=True,
                truncation=True,
                device=self.device
            )
            
    def _generate(self, prompt: str, system: str = "") -> str:
        return ollama.generate(self.llm_guard_model_name, prompt=prompt, system=system)['response']

    def _inference(self, text: str,) -> str:
        return self.encoder_model(text)[0]['label']
    
    @staticmethod
    def get_parse_llm_guard_output(model_output: str) -> dict[str, Optional[str]]:
        result = {}
        if 'unsafe' in model_output:
            result['category'] = model_output.split('\n')[-1]
            result['label'] = True
        else:
            result['category'] = None
            result['label'] = False
            
        return result

    @staticmethod
    def get_encoder_output(model_output: str) -> dict[str, str]:
        result = {"category": model_output}
        
        if model_output == 'toxic':
            result['label'] = True
        else:
            result['label'] = False
            
        return result
    
    def get_llm_guard_model_prediction(self, input_text: str) -> dict[str, Optional[str]]:
        return self.get_parse_llm_guard_output(self._generate(input_text))

    def get_encoder_model_prediction(self, input_text: str) -> dict[str, str]:
        return self.get_encoder_output(self._inference(input_text))
        
    
class OutputChecker:
    def __init__(
        self,
        output_detector: Optional[OutputDetector] = None,
        guard_model_type: Literal['llm', 'encoder'] = 'encoder',
        device: str = 'cpu'
    ) -> None:
        self.output_detector = output_detector if output_detector is not None else OutputDetector(
            guard_model_type=guard_model_type,
            device=device
        )
        self.guard_model_type = guard_model_type
        
    def get_llm_guard_model_label(self, input_text: str) -> tuple[Optional[str], bool]:
        llm_guard_model_prediction = self.output_detector.get_llm_guard_model_prediction(input_text)
        return llm_guard_model_prediction['category'], llm_guard_model_prediction['label']

    def get_encoder_model_label(self, input_text: str) -> tuple[Optional[str], bool]:
        encoder_model_prediction = self.output_detector.get_encoder_model_prediction(input_text)
        return encoder_model_prediction['category'], encoder_model_prediction['label']

    def check_output(self, input_text: str) -> tuple[Optional[str], bool]:
        if self.guard_model_type == 'llm':
            return self.get_llm_guard_model_label(input_text)
        elif self.guard_model_type == 'encoder':
            return self.get_encoder_model_label(input_text)
        