from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-960h", output_dim=None, dropout=0.3):
        super(ASRModel, self).__init__()
        
        # Load pretrained Wav2Vec2 model
        self.pretrained_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Optionally modify the output layer if needed
        if output_dim is not None:
            hidden_dim = self.pretrained_model.lm_head.in_features  # Get hidden size of the original model
            self.pretrained_model.lm_head = nn.Sequential(
                nn.Dropout(dropout), 
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x, attention_mask=None):
        # Remove the channel dimension and extract features
        x = x.squeeze(1)
        x = x.reshape(x.size(0), -1)  # Flatten the remaining dimensions to match [batch_size, sequence_length]
        
        # Pass the input through the pretrained model
        outputs = self.pretrained_model(input_values=x, attention_mask=attention_mask)
        
        return outputs.logits