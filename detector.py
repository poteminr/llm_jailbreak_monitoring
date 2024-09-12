from typing import Optional, Union

from detectors.input_checker.checker import InputChecker
from detectors.output_checker.checker import OutputChecker


class Detector:
    def __init__(
        self,
        input_checker: Optional[InputChecker] = None,
        output_checker: Optional[OutputChecker] = None,
        device: str = 'cpu'
    ) -> None:
        self.input_checker = input_checker if input_checker is not None else InputChecker(device=device)
        self.output_checker = output_checker if output_checker is not None else OutputChecker(device=device)
    
    @staticmethod
    def pack_input_check_results(
        input_text: str,
        input_check_results: tuple[float, bool]
    ) -> dict[str, Union[str, float, bool]]:
        results = {
            "input_text": input_text,
            "input_score": input_check_results[0],
            "is_input_jailbreak": input_check_results[1]
        }
        return results
        
    def check_input(self, input_text: str) -> dict[str, Union[str, float, bool]]:
        return self.pack_input_check_results(input_text, self.input_checker.check_input(input_text))
    
    @staticmethod
    def pack_output_check_results(
        generated_text: str,
        output_check_results: tuple[float, bool]
    ) -> dict[str, Union[Optional[str], bool]]:
        results = {
            "input_text": generated_text,
            "output_label": output_check_results[0],
            "is_unsafe": output_check_results[1]
        }
        return results
    
    def check_output(self, generated_text: str) -> dict[str, Union[Optional[str], bool]]:
        return self.pack_output_check_results(generated_text, self.output_checker.check_output(generated_text))
    
    @staticmethod
    def pack_results(
        input_text: str,
        generated_text: str,
        input_check_results: tuple[float, bool],
        output_check_results: tuple[Optional[str], bool]
        ) -> tuple[Optional[str], bool]:
        results = {
            "input_text" : input_text,
            "input_score": input_check_results[0],
            "is_input_jailbreak": input_check_results[1],
            "generated_text" : generated_text,
            "output_label": output_check_results[0],
            "is_unsafe": output_check_results[1]
        }
        return results
        
    def check_model_artefacts(
        self,
        input_text: str,
        generated_text: str
    ) -> dict[str, Union[float, bool, Optional[str]]]:
        input_check_results = self.input_checker.check_input(input_text)
        output_check_results = self.output_checker.check_output(generated_text)
        return self.pack_results(input_text, generated_text, input_check_results, output_check_results)