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
    def __init__(self, vocab_size, num_intent_labels, num_slot_labels, max_intents, nhead=4, d_model=128, num_layers=2):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,        # 词汇表大小，不会实用，就写个大点的数字
            hidden_size=d_model,         # Transformer的隐层维度
            num_hidden_layers=num_layers,     # Transformer Encoder层数
            num_attention_heads=nhead,   # 多头注意力个数
            intermediate_size=d_model*4,   # 前馈网络的中间层维度
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            # 其他参数可不改，默认即可
        )
        
        self.encoder = BertModel(config)
        self.intent_count_logits = nn.Linear(config.hidden_size, max_intents)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids) 
        # output.last_hidden_state: [batch_size, seq_len, hidden_size]
        
        pooled_output = outputs.pooler_output  # 最后一层 CLS 输出，[batch_size, hidden_size]
        sequence_output = outputs.last_hidden_state  # 最后一层每个token的隐藏状态，[batch_size, seq_len, hidden_size]
        
        intent_count_logits = self.intent_count_logits(pooled_output)  # [batch_size, max_intents]
        intent_logits = self.intent_classifier(pooled_output)  # [batch_size, num_intent_labels]
        slot_logits = self.slot_classifier(sequence_output)  # [batch_size, seq_len, num_slots]
        
        return intent_count_logits, intent_logits, slot_logits