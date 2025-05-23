# trainer.py

import torch
import numpy as np
from tqdm import tqdm
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_epoch(model, loader, optimizer, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, device, model_type="bert"):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        intent_labels = batch['intent_labels'].to(device)
        intent_counts = batch['intent_counts'].to(device)
        slot_labels = batch['slot_labels'].to(device)
        
        if model_type in ["bert", "transformer", "bert-freeze"]:
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            ic_logits, it_logits, slot_logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            ic_logits, it_logits, slot_logits = model(input_ids)
        
        # 假设意图数量已 1..N->0..N-1，否则可以保留 -1A
        loss_ic = intent_count_loss_fn(ic_logits, intent_counts - 1)
        loss_it = intent_loss_fn(it_logits, intent_labels)
        loss_sl = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
        
        loss = loss_ic + loss_it + loss_sl
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    print(f"训练损失: {avg_loss:.4f}")
    return avg_loss

def filter_label(label_str):
    # 保留所有标签，直接返回原标签
    return label_str

def evaluate(model, loader, device, intent_count_loss_fn, intent_loss_fn, slot_loss_fn, slots_num, model_type="bert"):
    model.eval()

    intent_count_preds, intent_count_true = [], []
    intent_preds, intent_true = [], []
    all_true_bio, all_pred_bio = [], []
    overall_correct = []  # 添加整体正确性统计

    # 反向映射
    id2slot = {v: k for k, v in slots_num.items()}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            intent_counts = batch['intent_counts'].to(device)
            intent_labels = batch['intent_labels'].to(device)
            slot_labels = batch['slot_labels'].to(device)

            if model_type in ["bert", "transformer", "bert-freeze"]:
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                ic_logits, it_logits, slot_logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                ic_logits, it_logits, slot_logits = model(input_ids)

            # 意图数量预测
            ic_preds_batch = torch.argmax(ic_logits, dim=1).cpu().numpy()
            ic_true_batch = (intent_counts.cpu().numpy() - 1)
            intent_count_preds.extend(ic_preds_batch)
            intent_count_true.extend(ic_true_batch)

            it_logits_cpu = it_logits.cpu().numpy()
            it_labels_cpu = intent_labels.cpu().numpy()
            slot_preds_batch = torch.argmax(slot_logits, dim=2).cpu().numpy()
            slot_labels_batch = slot_labels.cpu().numpy()
            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # === 意图数量 ===
                ic_pred = ic_preds_batch[i]
                ic_true = ic_true_batch[i]

                # === 意图标签预测 ===
                logits = it_logits_cpu[i]
                it_label = it_labels_cpu[i]
                N = ic_pred + 1
                if N > 0:
                    idxs = logits.argsort()[-N:][::-1]
                    pred_vec = np.zeros_like(logits)
                    pred_vec[idxs] = 1
                else:
                    pred_vec = np.zeros_like(logits)
                intent_preds.append(pred_vec)
                intent_true.append(it_label)

                # === 插槽 BIO ===
                t_seq = slot_labels_batch[i]
                p_seq = slot_preds_batch[i]
                t_bio, p_bio = [], []
                for t_id, p_id in zip(t_seq, p_seq):
                    if t_id == -100:
                        continue
                    t_label = id2slot.get(t_id, "O")
                    p_label = id2slot.get(p_id, "O")
                    t_bio.append(filter_label(t_label))
                    p_bio.append(filter_label(p_label))
                all_true_bio.append(t_bio)
                all_pred_bio.append(p_bio)

                # === Overall 正确性判断 ===
                ic_match = ic_pred == ic_true
                it_match = np.array_equal(pred_vec, it_label)
                slot_match = t_bio == p_bio
                overall_correct.append(int(ic_match and it_match and slot_match))

    # ====== 意图数量 ======
    ic_acc = accuracy_score(intent_count_true, intent_count_preds)
    ic_pre, ic_rec, ic_f1, _ = precision_recall_fscore_support(
        intent_count_true, intent_count_preds, average='macro', zero_division=0
    )

    # ====== 意图标签(多标签) ======
    it_true_np = np.array(intent_true)
    it_pred_np = np.array(intent_preds)
    it_acc = accuracy_score(it_true_np, it_pred_np)
    it_pre, it_rec, it_f1, _ = precision_recall_fscore_support(
        it_true_np, it_pred_np, average='macro', zero_division=0
    )

    # ====== 插槽 BIO（seqeval） ======
    print("\n--- seqeval classification_report ---")
    print(classification_report(all_true_bio, all_pred_bio, digits=4))

    slot_precision = precision_score(all_true_bio, all_pred_bio, average='macro')
    slot_recall = recall_score(all_true_bio, all_pred_bio, average='macro')
    slot_f1 = f1_score(all_true_bio, all_pred_bio, average='macro')

    # ====== Overall 指标 ======
    overall_acc = np.mean(overall_correct)
    print(f"\n=== Overall 准确率 ===")
    print(f"Overall Accuracy: {overall_acc:.4f}")

    # ====== 输出统计 ======
    print("\n=== 意图数量 ===")
    print(f"Acc: {ic_acc:.4f}, MacroP: {ic_pre:.4f}, MacroR: {ic_rec:.4f}, MacroF1: {ic_f1:.4f}")

    print("\n=== 意图识别 ===")
    print(f"Acc: {it_acc:.4f}, MacroP: {it_pre:.4f}, MacroR: {it_rec:.4f}, MacroF1: {it_f1:.4f}")

    print(f"\n=== 插槽填充 (seqeval, {model_type}) ===")
    print(f"MacroP: {slot_precision:.4f}, MacroR: {slot_recall:.4f}, MacroF1: {slot_f1:.4f}")

    return {
        'intent_count_accuracy': ic_acc,
        'intent_count_precision': ic_pre,
        'intent_count_recall': ic_rec,
        'intent_count_f1': ic_f1,
        'intent_accuracy': it_acc,
        'intent_precision': it_pre,
        'intent_recall': it_rec,
        'intent_f1': it_f1,
        'slot_precision': slot_precision,
        'slot_recall': slot_recall,
        'slot_f1': slot_f1,
        'overall_accuracy': overall_acc
    }
