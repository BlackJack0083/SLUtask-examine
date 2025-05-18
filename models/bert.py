import torch
from torch import nn
from transformers import BertModel, BertConfig
import os

class BertFreeze(nn.Module):
    def __init__(self, model_name, num_intent_labels, num_slot_labels, max_intents):
        super().__init__()
        
        # 获取当前文件 (bert.py) 的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在的目录 (models/)
        current_dir = os.path.dirname(current_file_path)
        # 获取当前文件所在目录的上级目录 (G:/)
        project_root = os.path.dirname(current_dir)
        # 构建 bert-base-cased 目录的绝对路径
        bert_model_path = os.path.join(project_root, 'bert-base-cased') 
        
        # 打印路径验证一下
        print(f"Resolved BERT model path: {bert_model_path}")
        
        self.bert = BertModel.from_pretrained(bert_model_path)  # 加载预训练的BERT模型
        
        hidden_size = self.bert.config.hidden_size  # 隐藏层大小
        
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结BERT模型参数，通过关闭梯度更新来实现
        
        self.intent_count_fc = nn.Linear(hidden_size, max_intents)
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)    
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)  # 输入BERT模型
        
        pooled_output = outputs.pooler_output  # 取出[CLS]输出，[batch_size, hidden_size]
        sequence_output = outputs.last_hidden_state # 取出每个token的隐藏状态，[batch_size, seq_len, hidden_size]
        
        intent_count_logits = self.intent_count_fc(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_count_logits, intent_logits, slot_logits
    
class Bert(nn.Module):
    def __init__(self, model_name, vocab_size, num_intent_labels, num_slot_labels, max_intents):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True) # 这里batch_first=True是为了适配TransformerEncoder输入格式
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)  # 6层Transformer编码器
        
        # 分类器
        self.intent_count_fc = nn.Linear(512, max_intents)
        self.intent_classifier = nn.Linear(512, num_intent_labels)
        self.slot_classifier = nn.Linear(512, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. 词嵌入
        embedded = self.embedding(input_ids)  # [batch, seq_len, 512]
        
        # 2. Transformer编码
        key_padding_mask = (attention_mask == 0)  # 转换为bool类型
        outputs = self.transformer_encoder(
            src=embedded,
            src_key_padding_mask=key_padding_mask
        )  # [batch, seq_len, 512]
        
        # 3. 意图分类（使用[CLS]）
        cls_output = outputs[:, 0, :]  # 第一个token
        intent_count_logits = self.intent_count_fc(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        
        # 4. 槽填充（全序列）
        slot_logits = self.slot_classifier(outputs)  # [batch, seq_len, num_slots]
        
        return intent_count_logits, intent_logits, slot_logits