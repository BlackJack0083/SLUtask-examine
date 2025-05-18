# data_preprocessing.py

import json
import torch
import os
from collections import defaultdict
from transformers import BertTokenizerFast

def load_txt(file_path):
    """加载ATIS/SNIPS格式的文本文件"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # 跳过空行
        while i < len(lines) and len(lines[i].strip()) == 0:
            i += 1
        if i >= len(lines):
            break
        
        tokens = []
        slots = []
        intents = []
        
        # 读取直到遇到意图行
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # 检查是否是意图行（没有空格的行）
            if len(line.split()) == 1:
                # 处理多意图，拆分意图并去除空格
                intents = [intent.strip() for intent in line.split('#')]  # 修改此处：按#分割
                # print(f"Found intents: {intents}")  # 打印每个样本的意图
                i += 1
                break
            
            parts = line.split()
            if len(parts) != 2:
                print(f"Ignoring invalid line: {line}")
                i += 1
                continue
                
            token, slot = parts
            tokens.append(token)
            slots.append(slot)
            i += 1
        
        # 有效性检查
        if len(tokens) != len(slots):
            print(f"忽略无效样本（token/slot数量不匹配）")
            continue
        if not intents:
            print(f"忽略无效样本（缺少意图）")
            continue
        
        samples.append({
            'tokens': tokens,
            'slots': slots,
            'intents': intents,
            'intent_count': len(intents)
        })
    
    return samples

def build_vocab(data_list):
    """自动构建slot和intent的词汇表"""
    slot_vocab = defaultdict(int)
    intent_vocab = defaultdict(int)
    
    for data in data_list:
        for sample in data:
            for slot in sample['slots']:
                slot_vocab[slot] += 1
            for intent in sample['intents']:
                intent_vocab[intent] += 1
    
    # 打印槽位词汇表内容，确保所有槽位类型都被收录
    print(f"Slot vocab: {slot_vocab}")
    
    # 确保'O'标签为0
    slots = ['O'] + [s for s in slot_vocab if s != 'O']
    slots_num = {s: i for i, s in enumerate(slots)}
    
    # 排序intents
    intents = sorted(intent_vocab.keys())
    intents_num = {i: idx for idx, i in enumerate(intents)}
    
    return slots_num, intents_num

def encode_data(data, tokenizer, intents_num, slots_num, max_len=128):
    """数据编码函数"""
    input_ids = []
    attention_masks = []
    intent_labels = []
    slot_labels = []
    intent_counts = []
    token_type_ids = []

    for item in data:
        # Tokenization
        encoding = tokenizer(
            item['tokens'],
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # 多标签意图编码
        intent_vec = torch.zeros(len(intents_num), dtype=torch.float)
        for intent in item['intents']:
            if intent in intents_num:
                intent_vec[intents_num[intent]] = 1
        intent_labels.append(intent_vec)
        intent_counts.append(len(item['intents']))

        # Slot编码
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(slots_num.get(item['slots'][word_idx], slots_num['O']))
                previous_word_idx = word_idx
            else:
                labels.append(-100)
        
        # 填充/截断到max_len
        if len(labels) < max_len:
            labels += [-100] * (max_len - len(labels))
        elif len(labels) > max_len:
            labels = labels[:max_len]
        
        # print("labels:", labels)

        slot_labels.append(torch.tensor(labels, dtype=torch.long))
        input_ids.append(encoding['input_ids'][0])
        
        attention_masks.append(encoding['attention_mask'][0])
        
        token_type_ids.append(encoding['token_type_ids'][0] if 'token_type_ids' in encoding else torch.zeros_like(encoding['input_ids'][0]))  

    return {
        'input_ids': torch.stack(input_ids),
        # 'attention_mask': torch.stack([torch.ones_like(i) for i in input_ids]),  # 简化处理
        'attention_mask': torch.stack(attention_masks),
        'token_type_ids': torch.stack(token_type_ids),
        'intent_labels': torch.stack(intent_labels),
        'slot_labels': torch.stack(slot_labels),
        'intent_counts': torch.tensor(intent_counts, dtype=torch.long)
    }

def main():
    # 文件路径配置
    train_file = 'MixATIS_clean/train.txt'
    val_file = 'MixATIS_clean/dev.txt'
    test_file = 'MixATIS_clean/test.txt'
    save_path = 'MixATIS_clean_processed.pth'

    # 加载原始数据
    train_data = load_txt(train_file)
    val_data = load_txt(val_file)
    test_data = load_txt(test_file)

    # 构建词汇表
    slots_num, intents_num = build_vocab([train_data, val_data, test_data])

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-cased')

    # 编码数据
    print("Encoding training data...")
    train_enc = encode_data(train_data, tokenizer, intents_num, slots_num)
    print("Encoding validation data...")
    val_enc = encode_data(val_data, tokenizer, intents_num, slots_num)
    print("Encoding test data...")
    test_enc = encode_data(test_data, tokenizer, intents_num, slots_num)

    # 保存处理结果
    torch.save({
        'train': train_enc,
        'val': val_enc,
        'test': test_enc,
        'slots_num': slots_num,
        'intents_num': intents_num
    }, save_path)

    print(f"处理完成！保存至：{save_path}")
    print(f"发现 {len(slots_num)} 种slot类型")
    print(f"发现 {len(intents_num)} 种意图类型")
    print(intents_num)
    print(slots_num)

if __name__ == "__main__":
    main()