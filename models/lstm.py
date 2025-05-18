import torch
from torch import nn

#######################################
# LSTM模型 (示例)
#######################################
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.intent_num_classifier = nn.Linear(hidden_dim*2, max_intents)
        self.intent_classifier = nn.Linear(hidden_dim*2, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_dim*2, num_slot_labels)

    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # 编码
        x, (h_n, c_n) = self.lstm(x)  # x的维度：[batch_size, seq_len, hidden_dim*2]， h_n和c_n的维度：[num_layers*num_directions, batch_size, hidden_dim]
        
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1) # 取最后一个时刻的隐层状态作为最后的隐层状态，维度：[batch_size, hidden_dim*2]
        
        # 分类
        ic_logits = self.intent_num_classifier(h_last)  # [batch_size, max_intents]
        it_logits = self.intent_classifier(h_last)  # [batch_size, num_intent_labels]
        
        slot_logits = self.slot_classifier(x)  # [batch_size, seq_len, num_slot_labels]

        return ic_logits, it_logits, slot_logits