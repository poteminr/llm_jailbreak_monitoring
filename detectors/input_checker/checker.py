from typing import Callable, Literal, Optional
import ollama

from detectors.input_checker.utils import LLM_GUARD_MODEL_PROMPT
from detectors.input_checker.base_guard_model import BaseGuardModel


class InputDetecor:
    def __init__(
        self,
        base_guard_model_name: str = "meta-llama/Prompt-Guard-86M",
        llm_guard_model_name: str = "llama3.1"
    ) -> None:
        self.base_guard_model = BaseGuardModel(base_guard_model_name)
        self.llm_guard_model_name = llm_guard_model_name
    
    def get_base_guard_model_score(self, input_text: str) -> tuple[float, float]:
        return self.base_guard_model.get_prompt_scores(input_text)
    
    def _generate_llm_injection_model(self, prompt: str, system: str = "") -> str:
        return ollama.generate(self.llm_guard_model_name, prompt=prompt, system=system)['response']

    def create_prompt(self, input_text: str, rag_context: str = "") -> str:
        return LLM_GUARD_MODEL_PROMPT.format(user_input=input_text, rag_context=rag_context)
    

class InputChecker:
    def __init__(
        self,
        find_sim_injection: Optional[Callable[[str, int], tuple[list[str], list[float]]]] = None,
        input_detector: Optional[InputDetecor] = None,
        base_guard_model_name: Optional[str] = None,
        llm_guard_model_name: Optional[str] = None,
        base_guard_model_score_threshold: float = 0.75,
        base_guard_model_scores_spread_threshold: float = 0.5,
        injection_search_similarity_threshold: float = 0.8,
        policy: Literal["simple", 'base', 'hard'] = 'base'
    ) -> None:
        
        self.find_sim_injection = find_sim_injection
    
        if base_guard_model_name is not None and llm_guard_model_name is not None:
            self.input_detector = InputDetecor(base_guard_model_name, llm_guard_model_name)
        else:
            self.input_detector = input_detector if input_detector is not None else InputDetecor()
    
        self.base_guard_model_score_threshold = base_guard_model_score_threshold
        self.base_guard_model_scores_spread_threshold = base_guard_model_scores_spread_threshold
        self.injection_search_similarity_threshold = injection_search_similarity_threshold
        self.policy = policy
        
    def get_base_guard_model_label(self, input_text: str) -> tuple[float, bool]:
        original_score, preprocessed_text_score = self.input_detector.get_base_guard_model_score(input_text)
        max_score = max(original_score, preprocessed_text_score)
        scores_spread = abs(preprocessed_text_score - original_score)
        
        if max_score >= self.base_guard_model_score_threshold:
            return max_score, True
        elif scores_spread >= self.base_guard_model_scores_spread_threshold:
            return scores_spread, True
        else:
            return max(max_score, scores_spread), False
