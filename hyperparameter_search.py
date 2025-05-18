# hyperparameter_search.py

import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import os
import time
import json

# 导入模型类
from models import (
    CNN,
    LSTM,
    GRU,
    RBFN,
    BertFreeze, # 对应 model_type='bertfreeze' (Frozen HF Bert)
    Bert,       # 对应 model_type='bert' (Hugging Face BertModel based on config)
    Transformer # 对应 model_type='transformer' (PyTorch TransformerEncoder based)
)

# 导入训练和评估函数
from trainer import train_epoch, evaluate # 确保 evaluate 函数已根据之前的建议修改为多标签评估

# 导入数据集类
from data.data_utils import SLUDataset


def run_experiment(model_type, params, device, train_loader, val_loader, intents_num, slots_num, max_intents, vocab_size, epochs): # 添加 vocab_size 参数
    """
    运行单次实验并在验证集上评估
    :param val_loader: 用于评估的验证集 DataLoader
    :param epochs: 当前实验训练的 epoch 数量
    :param vocab_size: 词汇表大小，用于非 BertFreeze 模型的 embedding 层
    """
    print(f"\n--- Running experiment with {model_type} and params: {params} ---")

    # 根据模型类型和参数初始化模型
    model_class_map = {
        'cnn': CNN,
        'lstm': LSTM,
        'gru': GRU,
        'rbfn': RBFN,
        'bertfreeze': BertFreeze,
        'bert': Bert,       # 使用 BertModel(config) 的类
        'transformer': Transformer # 使用 PyTorch TransformerEncoder 的类
    }

    model_class = model_class_map.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")

    # 准备模型初始化参数
    model_init_params = {}
    if model_type == 'cnn': # <-- 将 CNN 单独处理
        model_init_params = {
            'vocab_size': vocab_size,
            'embed_dim': params['embed_dim'],
            # 不传递 hidden_dim 给 CNN
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
    elif model_type in ['lstm', 'gru', 'rbfn']:
        model_init_params = {
            'vocab_size': vocab_size, # 传递 vocab_size
            'embed_dim': params['embed_dim'],
            'hidden_dim': params['hidden_dim'],
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }
    elif model_type == 'bertfreeze':
         model_init_params = {
            'model_name': params.get('model_name', 'bert-base-cased'),
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
            # BertFreeze 的维度参数不需要在此处传入，因为它从预训练模型加载
        }
    elif model_type == 'bert': # 使用 BertModel(config) 的类
        model_init_params = {
            # Bert 类现在需要这些参数
            # 注意参数名需要和你的 class Bert.__init__ 匹配
            'model_name': params.get('model_name', 'bert-base-cased'), # 虽然是 config init，但也保留 model_name 参数
            'd_model': params['dim'], # 'dim' -> d_model (hidden_size in BertConfig)
            'nhead': params['nhead'], # 'nhead' -> nhead (num_attention_heads in BertConfig)
            'num_layers': params['num_layers'], # 'num_layers' -> num_layers (num_hidden_layers in BertConfig)
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents,
            # 注意：如果你在 Bert 类中使用了 vocab_size 参数（比如 BertConfig 需要），需要在此处传递
            # 当前你的 Bert 类 BertConfig hardcode 了 vocab_size=30522，如果需要搜索或使用实际值，请修改类定义并传递
        }
    elif model_type == 'transformer': # 使用 PyTorch TransformerEncoder 的类
         model_init_params = {
            # Transformer 类现在需要这些参数
            # 注意参数名需要和你的 class Transformer.__init__ 匹配
            'model_name': params.get('model_name', 'transformer'), # 标识符
            'vocab_size': vocab_size, # 这个模型需要 vocab_size 来初始化 nn.Embedding
            'd_model': params['dim'], # 'dim' -> d_model
            'nhead': params['nhead'], # 'nhead' -> nhead
            'num_layers': params['num_layers'], # 'num_layers' -> num_layers
            'num_intent_labels': len(intents_num),
            'num_slot_labels': len(slots_num),
            'max_intents': max_intents
        }


    # 在实例化模型之前，检查 d_model / nhead 的整除性，尤其是对于 Bert 和 Transformer 类型
    if model_type in ['bert', 'transformer']:
         if params['dim'] % params['nhead'] != 0:
             print(f"Skipping combination due to dim ({params['dim']}) not divisible by nhead ({params['nhead']})")
             return None, None # 跳过此组合

    model = model_class(**model_init_params)
    model.to(device)

    # 定义损失函数
    intent_count_loss_fn = nn.CrossEntropyLoss()
    intent_loss_fn = nn.BCEWithLogitsLoss()
    slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 定义优化器，使用当前学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params['lr']))

    # 训练
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        # print(f"Epoch {epoch+1}/{epochs}") # 不打印太多，搜索过程会很长
        # 训练时，Bert类需要 attention_mask 和 token_type_ids
        if model_type in ['bertfreeze', 'bert', 'transformer']:
             train_epoch(model, train_loader, optimizer, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, device, model_type)
        else:
             # 其他模型可能只需要 input_ids
             train_epoch(model, train_loader, optimizer, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, device, model_type)


    # 评估 (在验证集上)
    print("Evaluating on validation set...")
    # 评估时，Bert类需要 attention_mask 和 token_type_ids
    if model_type in ['bertfreeze', 'bert', 'transformer']:
         eval_results = evaluate(model, val_loader, device, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, slots_num, model_type)
    else:
         # 其他模型可能只需要 input_ids
         eval_results = evaluate(model, val_loader, device, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, slots_num, model_type)


    return model, eval_results


def main():
    parser = argparse.ArgumentParser(description="SLU Hyperparameter Search")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to base config file')
    parser.add_argument('--model_type', type=str, required=True, choices=['cnn','lstm', 'gru', 'rbfn', 'bertfreeze', 'bert', 'transformer'],
                        help='Model type to search hyperparameters for')
    parser.add_argument('--search_epochs', type=int, default=10, help='Number of epochs to train for each hyperparameter combination')
    parser.add_argument('--save_dir', type=str, default='checkpoints/hyperparameter_search', help='Base directory to save best checkpoints')


    args = parser.parse_args()

    # 读取基础配置
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- 定义超参数搜索空间 ---
    param_grid = {
        'lr': [1e-4, 2e-4, 1e-3],

        # 通用密集网络和 RNN 模型参数 (CNN, LSTM, GRU, RBFN)
        'general': {
            'embed_dim': [32, 64],
            'hidden_dim': [64, 128],
        },

        # Transformer/Bert 类模型的参数 (Bert, Transformer)
        # 'dim' 对应 Bert 类的 d_model (hidden_size in BertConfig) 和 Transformer 类的 d_model
        # 'nhead' 对应 Bert 类的 nhead (num_attention_heads in BertConfig) 和 Transformer 类的 nhead
        # 'num_layers' 对应 Bert 类的 num_layers (num_hidden_layers in BertConfig) 和 Transformer 类的 num_layers
        'transformer_bert': {
            'dim': [256, 512],
            'nhead': [8],
            'num_layers': [2, 6], # 添加 num_layers 到搜索空间
        },
    }

    # --- 加载数据 ---
    data_path = cfg['data_path']
    print("加载预处理后的数据:", data_path)
    data = torch.load(data_path)
    print("数据加载完成")
    
    train_encodings = data['train']
    val_encodings = data['val'] # 加载验证集
    test_encodings = data['test'] # 虽然搜索不用，但可能需要其 stats
    intents_num = data['intents_num']
    slots_num = data['slots_num']
    # 确保 vocab_size 是实际使用的词汇表大小
    # 如果你的数据处理脚本已经确定了词汇表大小并保存在 data 字典中，优先使用它
    vocab_size = data.get('vocab_size', cfg.get('vocab_size', 30522))
    print(f"使用词汇表大小: {vocab_size}")


    # 拼接意图数量，找最大的意图数
    max_intents = max(train_encodings['intent_counts'].tolist() +
                      val_encodings['intent_counts'].tolist() +
                      test_encodings['intent_counts'].tolist())
    print(f"最大意图数量: {max_intents}")

    # 构建Dataset和DataLoader (批量大小从config获取)
    batch_size = cfg.get('batch_size', 16)
    train_dataset = SLUDataset(train_encodings)
    val_dataset = SLUDataset(val_encodings) # 创建验证集 Dataset
    # test_dataset = SLUDataset(test_encodings) # 测试集Dataset，搜索时不需要DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # 创建验证集 DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 测试集DataLoader，搜索时不需要

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # --- 设置当前模型类型的搜索参数 ---
    model_type = args.model_type

    best_overall_accuracy = -1.0
    best_params = None
    # best_model_state_dict = None # 不在此处保存 state_dict，在发现最优时临时保存

    # 设置当前模型类型的保存目录
    model_save_dir = os.path.join(args.save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    # 获取适用于当前模型类型的参数网格
    if model_type in ['cnn']:
        # CNN 模型搜索 lr 和 embed_dim
        current_param_grid_values = {
            'lr': param_grid['lr'],
            'embed_dim': param_grid['general']['embed_dim'] # embed_dim 来自 general 部分
        }
        param_names = ['lr', 'embed_dim']
    elif model_type in ['lstm', 'gru', 'rbfn']:
        current_param_grid_values = {**{'lr': param_grid['lr']}, **param_grid['general']}
        param_names = ['lr', 'embed_dim', 'hidden_dim']
    elif model_type in ['bert', 'transformer']:
         current_param_grid_values = {**{'lr': param_grid['lr']}, **param_grid['transformer_bert']}
         param_names = ['lr', 'dim', 'nhead', 'num_layers'] # 添加 num_layers 到 param_names
    elif model_type == 'bertfreeze':
        current_param_grid_values = {'lr': param_grid['lr']}
        param_names = ['lr']
    else:
        print(f"Error: Model type '{model_type}' is not defined in search grid.")
        return # 如果模型类型不在列表中则退出


    # 生成所有超参数组合
    param_combinations = list(itertools.product(*current_param_grid_values.values()))
    print(f"Total combinations for {model_type}: {len(param_combinations)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    start_time = time.time()

    for i, combo_values in enumerate(param_combinations):
        # 构建当前超参数字典
        current_params = dict(zip(param_names, combo_values))

        # 添加其他固定参数
        # BertFreeze 需要 model_name 参数
        current_params['model_name'] = cfg.get('model_name', 'bert-base-cased')
        # vocab_size 传递给 run_experiment

        print(f"\n--- Testing {model_type} Combination {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {current_params}")

        try:
            # 运行实验
            model, eval_results = run_experiment(
                model_type=model_type,
                params=current_params,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader, # 使用验证集 DataLoader
                intents_num=intents_num,
                slots_num=slots_num,
                max_intents=max_intents,
                vocab_size=vocab_size, # 传递 vocab_size
                epochs=args.search_epochs # 使用搜索指定的 epoch 数量
            )

            # 如果 run_experiment 返回 None (表示跳过组合)
            if model is None and eval_results is None:
                continue

            current_overall_accuracy = eval_results.get('overall_accuracy', -1.0) # 获取整体准确率

            print(f"Validation Overall Accuracy: {current_overall_accuracy:.4f}") # 明确是验证集结果

            # 比较并保存最优模型 (针对当前 model_type)
            if current_overall_accuracy > best_overall_accuracy:
                best_overall_accuracy = current_overall_accuracy
                # 复制一份当前参数，避免在后续循环中被修改
                best_params = current_params.copy()

                # 保存最优模型的 state_dict 到临时文件
                best_ckpt_path_temp = os.path.join(model_save_dir, f"best_temp_model_{model_type}.pth")
                torch.save(model.state_dict(), best_ckpt_path_temp)
                print(f">>> New Best Overall Accuracy Found: {best_overall_accuracy:.4f}")
                print(f">>> Corresponding Parameters: {best_params}")
                print(f">>> Best state_dict saved temporarily to {best_ckpt_path_temp}")


        except Exception as e:
             print(f"Error running experiment for {model_type} with params {current_params}: {e}")
             import traceback
             traceback.print_exc() # 打印完整的错误信息
             print("Skipping this combination.")


    end_time = time.time()
    search_duration = end_time - start_time

    # --- 当前模型类型搜索结束，保存最终最优结果 ---
    print(f"\n--- Search finished for model type: {model_type} ---")
    print(f"Total search duration for {model_type}: {search_duration:.2f} seconds")

    if best_params:
        # 将最终最优模型重命名为更清晰的名字
        final_best_ckpt_path = os.path.join(model_save_dir, f"best_model_{model_type}_overall_acc_{best_overall_accuracy:.4f}.pth")
        temp_ckpt_path = os.path.join(model_save_dir, f"best_temp_model_{model_type}.pth")
        if os.path.exists(temp_ckpt_path):
             # 只有当临时文件存在且保存成功时才执行重命名
             try:
                 os.rename(temp_ckpt_path, final_best_ckpt_path)
             except OSError as e:
                 print(f"Error renaming temporary checkpoint file: {e}")
                 final_best_ckpt_path = temp_ckpt_path # 如果重命名失败，也指向临时文件路径

        # 保存最佳参数到文件
        best_params_file = os.path.join(model_save_dir, f"best_params_{model_type}.json")
        try:
            with open(best_params_file, 'w') as f:
                # 将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
                serializable_params = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in best_params.items()}
                json.dump(serializable_params, f, indent=4)
            print(f"Best parameters saved to: {best_params_file}")
        except Exception as e:
             print(f"Error saving best parameters to JSON: {e}")


        print("\nBest Parameters Found:")
        print(json.dumps(best_params, indent=4)) # 这里可能仍然打印 numpy 类型
        print(f"\nBest Overall Accuracy on Validation Set: {best_overall_accuracy:.4f}")
        print(f"Best model checkpoint saved to: {final_best_ckpt_path}")

    else:
        print(f"\nNo successful experiments completed for model type: {model_type}.")
    print("="*50)


if __name__ == "__main__":
    main()