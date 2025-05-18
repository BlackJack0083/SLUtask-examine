import torch
from torch import nn

## RBFN
class SLU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents):
        super(SLU, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.intent_num_classifier = nn.Linear(hidden_dim*2, max_intents)
        self.intent_classifier = nn.Linear(hidden_dim*2, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_dim*2, num_slot_labels)

    def forward(self, x):
        # 编码
        x, (h_n, c_n) = self.lstm(x)
        
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        # 分类
        ic_logits = self.intent_num_classifier(h_last)
        it_logits = self.intent_classifier(h_last)
        
        slot_logits = self.slot_classifier(x)

        return ic_logits, it_logits, slot_logits

class RBFN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents):
        super(RBFN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 定义三个SLU模型
        self.slu1 = SLU(vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents)
        self.slu2 = SLU(vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents)
        self.slu3 = SLU(vocab_size, embed_dim, hidden_dim, num_intent_labels, num_slot_labels, max_intents)
        
        self.intent_embedding = nn.Linear(num_intent_labels, hidden_dim)
        self.slot_embedding = nn.Linear(num_slot_labels, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1, bias=False)
        self.slot_att = nn.Linear(hidden_dim, hidden_dim)
        
        self.new_intent = nn.Linear(hidden_dim + embed_dim, embed_dim)
        self.new_slot = nn.Linear(hidden_dim + embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # slu1输出
        ic_logits, it_logits1, slot_logits1 = self.slu1(embed)  # [batch_size, max_intents], [batch_size, num_intent_labels], [batch_size, seq_len, num_slot_labels]
        
        # 转化为隐式映射
        lsvI = self.intent_embedding(self.sigmoid(it_logits1))   # [batch_size, hidden_dim]
        ls = self.slot_embedding(self.softmax(slot_logits1))   # [batch_size, seq_len, hidden_dim]
        ls_att = self.softmax(self.Va(self.tanh(self.slot_att(ls))))  # [batch_size, seq_len, 1]
        lsvS = torch.sum(ls_att * ls, dim=1)  # [batch_size, hidden_dim]
        
        new_intent = self.new_intent(torch.cat((lsvI.unsqueeze(1).expand(-1, embed.size(1), -1), embed), dim=-1))  # [batch_size, seq_len, embed_dim]
        new_slot = self.new_slot(torch.cat((lsvS.unsqueeze(1).expand(-1, embed.size(1), -1), embed), dim=-1))  # [batch_size, seq_len, embed_dim]
        
        # slu2输出
        _, it_logits2, _ = self.slu2(new_slot)  # [batch_size, num_intent_labels], [batch_size, seq_len, num_slot_labels]
        
        # slu3输出
        _, _, slot_logits2 = self.slu3(new_intent)  # [batch_size, seq_len, num_slot_labels]
        
        intent_logits = (it_logits1 + it_logits2) / 2  # [batch_size, num_intent_labels]
        slot_logits = (slot_logits1 + slot_logits2) / 2  # [batch_size, seq_len, num_slot_labels]
        
        return ic_logits, intent_logits, slot_logits