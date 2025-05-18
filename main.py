# main.py

import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
from trainer import train_epoch, evaluate
from models import (
    CNN,
    LSTM,
    GRU,
    RBFN,
    BertFreeze,
    Bert,
    Transformer
)
from data.data_utils import SLUDataset  

def main():
    parser = argparse.ArgumentParser(description="SLU main entry")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['bert','cnn','lstm', 'rbfn', 'gru', 'transformer'])
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    data_path = cfg['data_path']
    model_name = cfg.get('model_name', 'bert-base-cased')
    batch_size = cfg.get('batch_size', 16)
    epochs = cfg.get('epochs', 3)
    lr = cfg.get('lr', 2e-5)
    # 根据需要加载更多超参数
    vocab_size = cfg.get('vocab_size', 30522)
    embed_dim = cfg.get('embed_dim', 128)
    hidden_dim = cfg.get('hidden_dim', 128)
    
    print("数据集：", data_path)
    print("加载预处理后的数据:", data_path)
    data = torch.load(data_path)
    print("数据加载完成")
    train_encodings = data['train']
    test_encodings = data['test']
    intents_num = data['intents_num']
    # print("意图数量:", intents_num)
    slots_num = data['slots_num']
    # print("槽位数量:", slots_num)
    
    # 拼接意图数量，找最大的意图数
    max_intents = max(train_encodings['intent_counts'].tolist() + test_encodings['intent_counts'].tolist())
    print(f"最大意图数量: {max_intents}")
    
    # 构建Dataset和DataLoader
    train_dataset = SLUDataset(train_encodings)
    test_dataset = SLUDataset(test_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
    
    # 根据 model_type 选择不同模型
    print("选择模型类型:", args.model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model_type == 'cnn':
        model = CNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    elif args.model_type == 'lstm':  # lstm
        model = LSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    elif args.model_type == 'gru':  # gru
        model = GRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    elif args.model_type == 'rbfn':
        model = RBFN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    elif args.model_type == 'bert':
        model = Bert(
            model_name=model_name,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
        # model = BertFreeze(
        #     model_name=model_name,
        #     num_intent_labels=len(intents_num),
        #     num_slot_labels=len(slots_num),
        #     max_intents=max_intents
        # )
    elif args.model_type == 'transformer':
        model = Transformer(
            model_name=model_name,
            vocab_size=vocab_size,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    else:
        model = GRU(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intent_labels=len(intents_num),
            num_slot_labels=len(slots_num),
            max_intents=max_intents
        )
    
    model.to(device)
    
    # 定义损失函数
    intent_count_loss_fn = nn.CrossEntropyLoss()
    intent_loss_fn = nn.BCEWithLogitsLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - {args.model_type}")
        train_epoch(model, train_loader, optimizer, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, device, args.model_type)
        evaluate(model, test_loader, device, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, slots_num, args.model_type)
    
    # 保存模型
    ckpt_path = f"checkpoints/{data_path.split('/')[-1].split('.')[0]}_{args.model_type}_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"模型已保存到 {ckpt_path}")

if __name__ == "__main__":
    main()
