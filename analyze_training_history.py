#!/usr/bin/env python3
"""
训练历史分析脚本
从 docs/*/train 目录中提取和分析训练历史数据
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def extract_task1_metrics(train_dir: Path) -> Dict[str, Any]:
    """从 task1 的训练历史中提取最终指标"""
    csv_file = train_dir / "synthetic_training_history.csv"

    if not csv_file.exists():
        return {}

    df = pd.read_csv(csv_file)
    last_epoch = df.iloc[-1]

    # 从列名中提取指标（注意中文字符）
    metrics = {
        "task": "Task 1 - 标签分类 (Label Classification)",
        "最终epoch": int(last_epoch["epoch"]),
        "train_loss": float(last_epoch["train_loss"]),
        "train_acc": float(last_epoch["train_acc"]),
        "train_recall": float(last_epoch["train_recall"]),
        "val_loss": float(last_epoch["val_loss"]),
        "val_acc": float(last_epoch["val_acc"]),
        "val_top5": float(last_epoch["val_top5"]),
        "val_macro_f1": float(last_epoch["val_macro_f1"]),
        "head_lr": float(last_epoch["head_lr"]),
        "backbone_lr": float(last_epoch["backbone_lr"]),
        "val_recall": float(last_epoch["val_recall"]),
    }

    return metrics


def extract_task2_metrics(train_dir: Path) -> Dict[str, Any]:
    """从 task2 的训练历史中提取最终指标"""
    csv_file = train_dir / "training_history.csv"

    if not csv_file.exists():
        return {}

    df = pd.read_csv(csv_file)
    last_epoch = df.iloc[-1]

    metrics = {
        "task": "Task 2 - 作物分类 (Crop Classification)",
        "最终epoch": int(last_epoch["epoch"]),
        "train_loss": float(last_epoch["train_loss"]),
        "train_acc": float(last_epoch["train_acc"]),
        "val_loss": float(last_epoch["val_loss"]),
        "val_acc": float(last_epoch["val_acc"]),
        "val_top5": float(last_epoch["val_top5"]),
        "val_macro_f1": float(last_epoch["val_macro_f1"]),
        "head_lr": float(last_epoch["head_lr"]),
        "backbone_lr": float(last_epoch["backbone_lr"]),
    }

    return metrics


def extract_task3_metrics(train_dir: Path) -> Dict[str, Any]:
    """从 task3 的训练历史中提取最终指标"""
    history_file = train_dir / "history.json"
    best_metrics_file = train_dir / "best_metrics.json"

    if not history_file.exists():
        return {}

    with open(history_file, "r") as f:
        history = json.load(f)

    # 最后一个epoch的指标
    last_idx = -1
    metrics = {
        "task": "Task 3 - 严重程度分类 (Severity Classification)",
        "最终epoch": int(history["epoch"][last_idx]),
        "train_loss": float(history["train_loss"][last_idx]),
        "train_acc": float(history["train_acc"][last_idx]),
        "val_loss": float(history["val_loss"][last_idx]),
        "val_acc": float(history["val_acc"][last_idx]),
        "val_macro_f1": float(history["val_macro_f1"][last_idx]),
        "learning_rate": float(history["learning_rate"][last_idx]),
    }

    # 如果有最佳指标文件，也读取它
    if best_metrics_file.exists():
        with open(best_metrics_file, "r") as f:
            best = json.load(f)
        metrics["best_val_acc"] = float(best["acc"])
        metrics["best_val_macro_f1"] = float(best["macro_f1"])
        metrics["best_val_loss"] = float(best["loss"])

        # 提取每个类别的召回率
        if "per_class_recall" in best:
            for class_name, recall in best["per_class_recall"].items():
                metrics[f"recall_{class_name}"] = float(recall)

    return metrics


def extract_task4_metrics(train_dir: Path) -> Dict[str, Any]:
    """从 task4 的训练历史中提取最终指标"""
    severity_file = train_dir / "severity_metrics_multitask.json"

    if not severity_file.exists():
        return {}

    with open(severity_file, "r") as f:
        data = json.load(f)

    metrics = {
        "task": "Task 4 - 多任务学习 (Multi-task Learning)",
    }

    # 直接从根级别提取严重程度分类的指标
    metrics["val_acc"] = float(data.get("accuracy", 0)) * 100  # 转换为百分比
    metrics["val_macro_f1"] = float(data.get("macro_f1", 0))

    # 从混淆矩阵计算召回率
    if "confusion_matrix" in data:
        cm = data["confusion_matrix"]
        # 假设类别顺序为 Healthy, Mild, Severe (3类)
        class_names = ["Healthy", "Mild", "Severe"]

        for i, class_name in enumerate(class_names):
            if i < len(cm):
                row_sum = sum(cm[i])
                if row_sum > 0:
                    recall = cm[i][i] / row_sum
                    metrics[f"recall_{class_name}"] = float(recall)

    return metrics


def fit_approximate_value(values: list, metric_name: str) -> float:
    """
    当某个指标缺失时，使用拟合方法估算一个近似值
    使用最后几个epoch的趋势进行线性外推
    """
    if not values or len(values) == 0:
        return 0.0

    # 如果只有一个值，直接返回
    if len(values) == 1:
        return float(values[0])

    # 使用最后5个点进行线性拟合
    n = min(5, len(values))
    x = np.arange(n)
    y = np.array(values[-n:])

    # 线性拟合
    coeffs = np.polyfit(x, y, 1)

    # 预测下一个值
    next_value = coeffs[0] * n + coeffs[1]

    return float(next_value)


def analyze_all_tasks():
    """分析所有任务的训练历史"""
    base_dir = Path("docs")

    results = {}

    # Task 1
    task1_dir = base_dir / "task1" / "train"
    if task1_dir.exists():
        print("正在分析 Task 1...")
        results["task1"] = extract_task1_metrics(task1_dir)

    # Task 2
    task2_dir = base_dir / "task2" / "train"
    if task2_dir.exists():
        print("正在分析 Task 2...")
        results["task2"] = extract_task2_metrics(task2_dir)

    # Task 3
    task3_dir = base_dir / "task3" / "train"
    if task3_dir.exists():
        print("正在分析 Task 3...")
        results["task3"] = extract_task3_metrics(task3_dir)

    # Task 4
    task4_dir = base_dir / "task4" / "train"
    if task4_dir.exists():
        print("正在分析 Task 4...")
        results["task4"] = extract_task4_metrics(task4_dir)

    return results


def print_summary(results: Dict[str, Dict[str, Any]]):
    """打印摘要信息"""
    print("\n" + "=" * 80)
    print("训练历史最终指标汇总")
    print("=" * 80 + "\n")

    for task_key, metrics in results.items():
        if not metrics:
            continue

        print(f"\n{'=' * 60}")
        print(f"{metrics.get('task', task_key)}")
        if task_key == "task3":
            print("⚠️  配置错误：实际为3类但训练时配置为4类")
        print(f"{'=' * 60}\n")

        # 按类别分组显示指标
        print("【基本信息】")
        if "最终epoch" in metrics:
            print(f"  最终 Epoch: {metrics['最终epoch']}")

        print("\n【训练指标】")
        if "train_loss" in metrics:
            print(f"  Train Loss: {metrics['train_loss']:.6f}")
        if "train_acc" in metrics:
            print(f"  Train Accuracy: {metrics['train_acc']:.2f}%")
        if "train_recall" in metrics:
            print(f"  Train Recall: {metrics['train_recall']:.2f}%")

        print("\n【验证指标】")
        if "val_loss" in metrics:
            print(f"  Val Loss: {metrics['val_loss']:.6f}")
        if "val_acc" in metrics:
            print(f"  Val Accuracy: {metrics['val_acc']:.2f}%")
        if "val_top5" in metrics:
            print(f"  Val Top-5 Accuracy: {metrics['val_top5']:.2f}%")
        if "val_macro_f1" in metrics:
            print(f"  Val Macro F1: {metrics['val_macro_f1']:.6f}")
        if "val_recall" in metrics:
            print(f"  Val Recall: {metrics['val_recall']:.2f}%")

        print("\n【最佳指标】(如果有)")
        if "best_val_acc" in metrics:
            print(f"  Best Val Accuracy: {metrics['best_val_acc']:.2f}%")
        if "best_val_macro_f1" in metrics:
            print(f"  Best Val Macro F1: {metrics['best_val_macro_f1']:.6f}")
        if "best_val_loss" in metrics:
            print(f"  Best Val Loss: {metrics['best_val_loss']:.6f}")

        print("\n【学习率】")
        if "head_lr" in metrics:
            print(f"  Head Learning Rate: {metrics['head_lr']:.9f}")
        if "backbone_lr" in metrics:
            print(f"  Backbone Learning Rate: {metrics['backbone_lr']:.9f}")
        if "learning_rate" in metrics:
            print(f"  Learning Rate: {metrics['learning_rate']:.9f}")

        print("\n【召回率详情】")
        recall_keys = [
            k
            for k in metrics.keys()
            if "recall" in k.lower() and k != "val_recall" and k != "train_recall"
        ]
        if recall_keys:
            for key in sorted(recall_keys):
                class_name = key.replace("recall_", "").replace("severity_recall_", "")
                print(f"  {class_name}: {metrics[key]:.4f}")


def save_to_csv(
    results: Dict[str, Dict[str, Any]], output_file: str = "training_final_metrics.csv"
):
    """将结果保存为CSV文件"""
    if not results:
        print("没有结果可保存")
        return

    # 收集所有可能的列名
    all_keys = set()
    for metrics in results.values():
        all_keys.update(metrics.keys())

    # 移除 'task' 键，稍后单独处理
    all_keys.discard("task")
    all_keys = ["task"] + sorted(all_keys)

    # 写入CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()

        for task_key, metrics in sorted(results.items()):
            if metrics:
                writer.writerow(metrics)

    print(f"\n结果已保存到: {output_file}")


def main():
    print("开始分析训练历史...")
    results = analyze_all_tasks()

    if results:
        print_summary(results)
        save_to_csv(results)
    else:
        print("未找到任何训练历史数据")


if __name__ == "__main__":
    main()
