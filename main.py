# main.py

import torch
import argparse
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import os # 导入 os 用于处理路径

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
    parser.add_argument('--model_type', type=str, default='transformer', # 默认模型类型
                        choices=['cnn','lstm', 'gru', 'rbfn', 'bert-freeze', 'bert', 'transformer'],
                        help='Model type to use')
    parser.add_argument('--data_path', type=str, help='Path to preprocessed data (overrides config)')

    # === 添加可以通过命令行修改的超参数 ===
    parser.add_argument('--model_name', type=str, help='Pre-trained model name for BertFreeze/Bert (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')

    # 通用模型参数 (CNN, LSTM, GRU, RBFN)
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension (overrides config)')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension for RNNs/RBFN (overrides config)')

    # Transformer/Bert 模型参数 (Bert, Transformer)
    parser.add_argument('--d_model', type=int, help='Model dimension for Transformers (overrides config)') # 对应 dim/hidden_size
    parser.add_argument('--nhead', type=int, help='Number of attention heads for Transformers (overrides config)')
    parser.add_argument('--num_layers', type=int, help='Number of encoder layers for Transformers (overrides config)')
    # === 超参数添加结束 ===

    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # === 从命令行参数或配置中确定最终超参数值 ===
    # 命令行参数优先
    data_path = args.data_path if args.data_path else cfg['data_path']
    model_name = args.model_name if args.model_name else cfg.get('model_name', 'bert-base-cased') # BertFreeze/Bert 可能需要 model_name
    batch_size = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 16)
    epochs = args.epochs if args.epochs is not None else cfg.get('epochs', 3)
    lr = args.lr if args.lr is not None else cfg.get('lr', 2e-5)

    # 通用维度参数 (先从 args 取，如果 args 为 None 再从 config 取，config 也没有则用默认值)
    embed_dim = args.embed_dim if args.embed_dim is not None else cfg.get('embed_dim', 128)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg.get('hidden_dim', 128)

    # Transformer/Bert 特定参数 (先从 args 取，如果 args 为 None 再从 config 取，config 也没有则用默认值)
    # 注意：Bert 和 Transformer 类可能在 __init__ 中有自己的默认值，这里的默认值只是用于统一展示和传递
    d_model = args.d_model if args.d_model is not None else cfg.get('d_model', 512)
    nhead = args.nhead if args.nhead is not None else cfg.get('nhead', 8)
    num_layers = args.num_layers if args.num_layers is not None else cfg.get('num_layers', 6)

    # vocab_size 通常由数据决定，但如果 config 有也可以用
    # 更好的方式是从加载的数据中获取 vocab_size
    # 先加载数据以获取 vocab_size 和 label 数量
    print("加载预处理后的数据:", data_path)
    data = torch.load(data_path)
    print("数据加载完成")

    train_encodings = data['train']
    # 如果数据处理脚本将 vocab_size 保存在 data 字典中，优先使用它
    vocab_size = data.get('vocab_size', cfg.get('vocab_size', 30522))
    print(f"使用词汇表大小: {vocab_size}")

    # 从数据中获取 label 数量
    intents_num = data['intents_num']
    slots_num = data['slots_num']
    # print("意图数量:", intents_num) # 可以在 evaluate 中打印详情
    # print("槽位数量:", slots_num) # 可以在 evaluate 中打印详情

    # 拼接意图数量，找最大的意图数 (用于意图数量分类器的输出维度)
    # 需要同时考虑训练集、验证集、测试集的最大意图数量，以防数据划分导致某个集合的最大值不是全局最大
    # 假设 data 字典中包含了 train, val, test 的 encodings
    all_intent_counts = []
    if 'train' in data and 'intent_counts' in data['train']:
        all_intent_counts.extend(data['train']['intent_counts'].tolist())
    if 'val' in data and 'intent_counts' in data['val']:
        all_intent_counts.extend(data['val']['intent_counts'].tolist())
    if 'test' in data and 'intent_counts' in data['test']:
        all_intent_counts.extend(data['test']['intent_counts'].tolist())

    max_intents = max(all_intent_counts) if all_intent_counts else 1 # 至少有1个意图数
    print(f"最大意图数量: {max_intents}")


    print("\n最终确定的参数：")
    print(f"模型类型: {args.model_type}")
    print(f"数据路径: {data_path}")
    print(f"模型名称 (Bert): {model_name}") # 打印 Bert 相关的 model_name
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"词汇表大小: {vocab_size}")
    print(f"嵌入维度 (通用): {embed_dim}") # 打印通用模型的维度
    print(f"隐藏维度 (RNN/RBFN): {hidden_dim}") # 打印 RNN/RBFN 的维度
    print(f"模型维度 (Transformer/Bert): {d_model}") # 打印 Transformer/Bert 的维度
    print(f"注意力头数 (Transformer/Bert): {nhead}") # 打印 Transformer/Bert 的头数
    print(f"编码器层数 (Transformer/Bert): {num_layers}") # 打印 Transformer/Bert 的层数
    print(f"意图标签数量: {len(intents_num)}")
    print(f"槽位标签数量: {len(slots_num)}")
    print(f"最大意图数量类别: {max_intents}") # 用于分类器的实际输出维度是 max_intents

    # 构建Dataset和DataLoader
    # 测试集也需要DataLoader，因为 evaluate 函数需要
    test_encodings = data['test'] # 重新获取 test_encodings (已在上面加载过 data)
    train_dataset = SLUDataset(train_encodings)
    test_dataset = SLUDataset(test_encodings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")


    # 根据 model_type 选择不同模型并传递参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = None # 初始化模型变量
    model_init_params = {} # 用于存储传递给模型 __init__ 的参数

    if args.model_type == 'cnn':
        model_init_params = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim, # 使用确定后的 embed_dim
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
        model = CNN(**model_init_params)

    elif args.model_type == 'lstm':
        model_init_params = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim, # 使用确定后的 embed_dim
            'hidden_dim': hidden_dim, # 使用确定后的 hidden_dim
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
        model = LSTM(**model_init_params)

    elif args.model_type == 'gru':
        model_init_params = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim, # 使用确定后的 embed_dim
            'hidden_dim': hidden_dim, # 使用确定后的 hidden_dim
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
        model = GRU(**model_init_params)

    elif args.model_type == 'rbfn':
        model_init_params = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim, # 使用确定后的 embed_dim
            'hidden_dim': hidden_dim, # 使用确定后的 hidden_dim
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
        model = RBFN(**model_init_params)

    elif args.model_type == 'bert-freeze':
        model_init_params = {
            'model_name': model_name, # BertFreeze 使用 model_name 加载预训练模型
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
            # BertFreeze 的维度和层数由加载的 model_name 决定
        }
        model = BertFreeze(**model_init_params)

    elif args.model_type == 'bert': # Hugging Face BertModel(config) based
        # 检查 d_model 和 nhead 是否兼容
        if d_model % nhead != 0:
            print(f"Error: For BertModel(config), d_model ({d_model}) must be divisible by nhead ({nhead}).")
            return # 退出程序

        model_init_params = {
            'd_model': d_model, # 使用确定后的 d_model (作为 hidden_size)
            'nhead': nhead, # 使用确定后的 nhead (作为 num_attention_heads)
            'num_layers': num_layers, # 使用确定后的 num_layers (作为 num_hidden_layers)
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents,
            # vocab_size 硬编码在 Bert 类中，如果需要外部控制，请修改 Bert 类
        }
        model = Bert(**model_init_params)

    elif args.model_type == 'transformer': # PyTorch TransformerEncoder based
         # 检查 d_model 和 nhead 是否兼容
        if d_model % nhead != 0:
            print(f"Error: For PyTorch TransformerEncoder, d_model ({d_model}) must be divisible by nhead ({nhead}).")
            return # 退出程序

        model_init_params = {
            'vocab_size': vocab_size, # Transformer 需要 vocab_size 用于 embedding
            'd_model': d_model, # 使用确定后的 d_model
            'nhead': nhead, # 使用确定后的 nhead
            'num_layers': num_layers, # 使用确定后的 num_layers
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
        model = Transformer(**model_init_params)

    else:
        print(f"Error: Model type '{args.model_type}' not supported.")
        return # 退出程序，避免 model 未被定义

    model.to(device)

    # 定义损失函数
    intent_count_loss_fn = nn.CrossEntropyLoss()
    intent_loss_fn = nn.BCEWithLogitsLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr)) # 使用确定后的 lr

    # 训练循环
    for epoch in range(epochs): # 使用确定后的 epochs
        print(f"\nEpoch {epoch+1}/{epochs} - {args.model_type}")
        train_epoch(model, train_loader, optimizer, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, device, args.model_type)
        evaluate(model, test_loader, device, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, slots_num, args.model_type)

    # 保存模型
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True) # 确保 checkpoint 目录存在
    # 构建一个包含关键参数的文件名
    param_str = f"{args.model_type}_lr{lr}_bs{batch_size}_ep{epochs}"
    if args.model_type in ['cnn']:
         param_str += f"_emb{embed_dim}"
    elif args.model_type in ['lstm', 'gru', 'rbfn']:
         param_str += f"_emb{embed_dim}_hid{hidden_dim}"
    elif args.model_type in ['bert', 'transformer']:
         param_str += f"_dim{d_model}_nhead{nhead}_lyr{num_layers}"
         if args.model_type == 'bert-freeze':
              param_str += f"_{model_name.replace('/', '-')}" # 替换斜杠避免路径问题

    data_filename = os.path.basename(data_path).split('.')[0] # 获取数据文件名部分
    ckpt_path = os.path.join(ckpt_dir, f"{data_filename}_{param_str}_model.pth")

    torch.save(model.state_dict(), ckpt_path)
    print(f"\n模型已保存到 {ckpt_path}")

if __name__ == "__main__":
    main()