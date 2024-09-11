import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional


class ToxicLLMClassifier(nn.Module, 
                         PyTorchModelHubMixin, 
                         repo_url="toxic_llm_classifier",
                         pipeline_tag="text_classifier",
                         license="mit"):
    def __init__(self, n_classes):
        super(ToxicLLMClassifier, self).__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        return self.classifier(output)

    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return preds

def infer(text: str,
          model: Optional[ToxicLLMClassifier] = ToxicLLMClassifier.from_pretrained("nikiduki/ToxicLLMClassifier"),
          tokenizer: Optional[DistilBertTokenizer] = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces=True),
          device: str = "cuda"
          )-> bool:
    """
    Model inference function that determines whether the text is toxic or not.
    :param model_name: name of classification model.
    :param tokenizer_name: name of tokenizer.
    :param device: name of the device where the model will be run.
    :return: boolean value of model toxicity.
    """
    
    model.to(device)
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model.predict(input_ids, attention_mask)
    
    return output.item()
