import torch
from torch import nn

#######################################
# CNN模型 (示例)
#######################################

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_intent_labels, num_slot_labels, max_intents):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)  # 池化层
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.intent_num_classifier = nn.Linear(embed_dim, max_intents)
        self.intent_classifier = nn.Linear(embed_dim, num_intent_labels)
        self.slot_classifier = nn.Linear(embed_dim, num_slot_labels)  
        
    def forward(self, x):
        
        # 词嵌入
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        
        # 卷积
        out_1 = self.conv1(x)  # [batch_size, embed_dim, seq_len]
        out_2 = self.conv2(out_1)  # [batch_size, embed_dim * 2, seq_len]
        out_3 = self.conv3(out_2)  # [batch_size, embed_dim, seq_len]
        
        # 池化
        pooled = self.pool(out_3)  # [batch_size, embed_dim, 1]
        result = pooled.squeeze(-1)  # [batch_size, embed_dim]
        
        c_out_t = out_3.permute(0, 2, 1)  # [batch_size, seq_len, embed_dim]
        
        # 分类
        ic_logits = self.intent_num_classifier(result)  # [batch_size, max_intents]
        it_logits = self.intent_classifier(result)   # [batch_size, num_intent_labels]
        slot_logits = self.slot_classifier(c_out_t)  # [batch_size, seq_len, num_slot_labels]
        
        return ic_logits, it_logits, slot_logits
        