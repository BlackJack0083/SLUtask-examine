# data_utils.py

import torch
from torch.utils.data import Dataset

class SLUDataset(Dataset):
    """
    专门的Dataset，用于封装编码后的输入数据:
      input_ids, attention_mask, token_type_ids, intent_labels, intent_counts, slot_labels
    """
    def __init__(self, encodings):
        """
        encodings 是一个字典，里面至少包含:
          {
            'input_ids': ...,
            'attention_mask': ...,
            'token_type_ids': ...,
            'intent_labels': ...,
            'intent_counts': ...,
            'slot_labels': ...
          }
        """
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        # 取单条样本的数据
        item = {key: self.encodings[key][idx] for key in 
                ['input_ids','attention_mask','token_type_ids']}
        item['intent_labels'] = self.encodings['intent_labels'][idx]
        item['intent_counts'] = self.encodings['intent_counts'][idx]
        item['slot_labels'] = self.encodings['slot_labels'][idx]
        return item
