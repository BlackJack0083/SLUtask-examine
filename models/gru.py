import torch
from torch import nn

class GRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.intent_num_classifier = nn.Linear(hidden_dim*2, max_intents)
        self.intent_classifier = nn.Linear(hidden_dim*2, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_dim*2, num_slot_labels)

    def forward(self, x):
        
        # 词嵌入
        x = self.embedding(x)
        
        # 编码
        x, h_n = self.gru(x)
        
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        # 分类
        ic_logits = self.intent_num_classifier(h_last)
        it_logits = self.intent_classifier(h_last)
        
        slot_logits = self.slot_classifier(x)

        return ic_logits, it_logits, slot_logits
