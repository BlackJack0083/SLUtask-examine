import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, model_name, vocab_size, num_intent_labels, num_slot_labels, max_intents, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 6层Transformer编码器
        
        # 分类器
        self.intent_count_fc = nn.Linear(d_model, max_intents)
        self.intent_classifier = nn.Linear(d_model, num_intent_labels)
        self.slot_classifier = nn.Linear(d_model, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. 词嵌入
        embedded = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # 2. Transformer编码
        key_padding_mask = (attention_mask == 0)  # 转换为bool类型
        outputs = self.transformer_encoder(
            src=embedded,
            src_key_padding_mask=key_padding_mask
        )  # [batch, seq_len, d_model]
        
        # 3. 意图分类（使用[CLS]）
        cls_output = outputs[:, 0, :]  # 第一个token
        intent_count_logits = self.intent_count_fc(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        
        # 4. 槽填充（全序列）
        slot_logits = self.slot_classifier(outputs)  # [batch, seq_len, num_slots]
        
        return intent_count_logits, intent_logits, slot_logits