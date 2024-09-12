import torch
import torch.nn as nn
from transformers import DistilBertModel
from huggingface_hub import PyTorchModelHubMixin


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
