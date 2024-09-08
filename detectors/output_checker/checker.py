from typing import Optional
import ollama

class OutputDetector:
    def __init__(self, llm_guard_model_name: str = "xe/llamaguard3") -> None:
        self.llm_guard_model_name = llm_guard_model_name
    
    def _generate(self, prompt: str, system: str = "") -> str:
        return ollama.generate(self.llm_guard_model_name, prompt=prompt, system=system)['response']
    
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
    
    def get_llm_guard_model_prediction(self, input_text: str) -> dict[str, Optional[str]]:
        return self.get_parse_llm_guard_output(self._generate(input_text))
        
    
class OutputChecker:
    def __init__(
        self,
        output_detector: Optional[OutputDetector] = None
    ) -> None:
        self.output_detector = output_detector if output_detector is not None else OutputDetector()
    
    def get_llm_guard_model_label(self, input_text: str) -> tuple[Optional[str], bool]:
        llm_guard_model_prediction = self.output_detector.get_llm_guard_model_prediction(input_text)
        return llm_guard_model_prediction['category'], llm_guard_model_prediction['label']

    def check_output(self, input_text: str) -> tuple[Optional[str], bool]:
        return self.get_llm_guard_model_label(input_text)