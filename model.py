import librosa

import torch
from transformers import AutoProcessor, AutoModelForPreTraining


class BaseRecognizer(torch.nn.Module):
    def __init__(self, model, feature_size, vocab_size):
        super().__init__()
        self.head = torch.nn.Linear(feature_size, vocab_size)
        self.processor = AutoProcessor.from_pretrained(model)

    def _get_features(self, input_values, attention_mask):
        raise NotImplementedError

    def get_feat_length(self, input_length):
        raise NotImplementedError

    def forward(self, inputs):
        features = self._get_features(inputs)
        logits = self.head(features)
        return logits


class Wav2Vec2Recognizer(BaseRecognizer):
    def __init__(self, model, vocab_size):
        feature_size = {
            "facebook/wav2vec2-base": 256,
            "facebook/wav2vec2-xls-r-300m": 1024,
        }[model]
        super().__init__(model, feature_size, vocab_size)
        self.net = AutoModelForPreTraining.from_pretrained(model)
        self.get_feat_length = self.net._get_feat_extract_output_lengths

    def freeze_conv_features(self):
        self.net.freeze_feature_encoder()

    def _get_features(self, inputs):
        outputs = self.net(inputs)
        return outputs[0]


class Wav2Vec2ConvRecognizer(BaseRecognizer):
    def __init__(self, model, vocab_size):
        feature_size = {
            "facebook/wav2vec2-base": 512,
            "facebook/wav2vec2-xls-r-300m": 512,
        }[model]
        super().__init__(model, feature_size, vocab_size)
        net = AutoModelForPreTraining.from_pretrained(model)
        self.extractor = net.wav2vec2.feature_extractor
        self.projector = net.wav2vec2.feature_projection
        self.get_feat_length = net._get_feat_extract_output_lengths

    def _get_features(self, inputs):
        feats = self.extractor(inputs)
        feats = feats.transpose(1, 2)
        _, feats = self.projector(feats)
        return feats

    def freeze_conv_features(self):
        for module in (self.extractor, self.projector):
            for param in module.parameters():
                param.requires_grad = False
