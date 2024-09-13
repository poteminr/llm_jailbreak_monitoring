from typing import Literal, Optional, Union
import ollama
from detectors.input_checker.utils import LLM_GUARD_MODEL_PROMPT
from detectors.input_checker.base_guard_model import BaseGuardModel

from vector_search.injection_finder import InjectionFinder


class InputDetector:
    def __init__(
        self,
        base_guard_model_name: str = "meta-llama/Prompt-Guard-86M",
        llm_guard_model_name: str = "llama3.1",
        llm_guard_model_prompt_template: str = LLM_GUARD_MODEL_PROMPT,
        device: str = "cpu"
    ) -> None:
        self.base_guard_model = BaseGuardModel(base_guard_model_name, device)
        self.llm_guard_model_name = llm_guard_model_name
        self.llm_guard_model_prompt_template = llm_guard_model_prompt_template
    
    def get_base_guard_model_score(self, input_text: str) -> tuple[float, float]:
        return self.base_guard_model.get_prompt_scores(input_text)
    
    def _generate(self, prompt: str, system: str = "") -> str:
        return ollama.generate(self.llm_guard_model_name, prompt=prompt, system=system)['response']

    def _create_prompt(self, input_text: str, rag_context: str = "") -> str:
        return self.llm_guard_model_prompt_template.format(user_input=input_text, rag_context=rag_context)
    
    def generate_llm_injection_model(self, input_text: str, rag_context: str = "") -> str:
        model_input = self._create_prompt(input_text, rag_context)
        return self._generate(prompt=model_input)


class InputChecker:
    def __init__(
        self,
        input_detector: Optional[InputDetector] = None,
        injection_finder: Optional[InjectionFinder] = None,
        injection_finder_model_name: str = "intfloat/multilingual-e5-large",
        injection_finder_embeddings_path: Optional[str] = None,
        base_guard_model_name: Optional[str] = None,
        llm_guard_model_name: Optional[str] = None,
        base_guard_model_score_threshold: float = 0.75,
        base_guard_model_scores_spread_threshold: float = 0.5,
        injection_search_similarity_threshold: float = 0.8,
        policy: Literal["simple", 'base', 'hard'] = 'base',
        device: str = 'cpu'
    ) -> None:
        if injection_finder_embeddings_path is not None:
            self.injection_finder = InjectionFinder(
                model_name_or_path=injection_finder_model_name,
                path_to_embeddings=injection_finder_embeddings_path,
                device=device
            )
        else:
            self.injection_finder = injection_finder if injection_finder is not None else InjectionFinder(device=device)
            
        if base_guard_model_name is not None and llm_guard_model_name is not None:
            self.input_detector = InputDetector(base_guard_model_name, llm_guard_model_name, device=device)
        else:
            self.input_detector = input_detector if input_detector is not None else InputDetector(device=device)
    
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
    
    def get_search_similarity_label(
        self,
        input_text: str,
        top_k: int = 1
    ) -> tuple[float, bool, tuple[list[str], list[float]]]:
        search_results = self.injection_finder.find_sim_injection(input_text, top_k)
        score = max(search_results[1])
        label = (score > self.injection_search_similarity_threshold)
        return score, label, search_results
    
    def get_llm_guard_model_label(
        self,
        input_text: str,
        search_results: Optional[tuple[list[str], list[float]]] = None,
        top_k: int = 1
    ) -> tuple[float, bool]:
        if search_results is None:
            search_results = self.get_search_similarity_label(input_text, top_k)
        
        rag_context = "\n".join(search_results[0])
        llm_guard_output = self.input_detector.generate_llm_injection_model(input_text, rag_context)
        return 0, False # TODO: parse output of llm guard model
        
    def _check_input_one_sample(self, input_text: str, top_k: int = 1) -> tuple[float, bool]:
        base_guard_model_label = self.get_base_guard_model_label(input_text)
        if self.policy == 'simple' or base_guard_model_label[1]:
            return base_guard_model_label
        
        search_similarity_label = self.get_search_similarity_label(input_text, top_k)
        if self.policy == 'base' or search_similarity_label[1]:
            return search_similarity_label[0], search_similarity_label[1]
        
        llm_guard_model_label = self.get_llm_guard_model_label(input_text, search_similarity_label[2], top_k)
        return llm_guard_model_label
    
    def check_input(
        self,
        input_text: Union[list[str], str],
        top_k: int = 1
    ) -> Union[tuple[float, bool], tuple[list[float], list[bool]]]:
        if isinstance(input_text, str):
            return self._check_input_one_sample(input_text, top_k)
        else:
            scores, labels = [], []
            for text in input_text:
                score, label = self._check_input_one_sample(text, top_k)
                scores.append(score)
                labels.append(label)
            return scores, labels