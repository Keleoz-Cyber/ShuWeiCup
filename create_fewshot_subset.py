#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Few-Shot (N-shot per class) Training Metadata for Task 2
================================================================

Task 2 要求：每个类别仅使用 10 张图进行训练（61 类共最多 610 张样本），禁止额外训练数据。
本脚本从完整训练 metadata 中，为每个类别随机抽取 N 张（默认 10）样本，生成 few-shot 训练用的 metadata CSV。

特性：
- 固定每类样本数（默认 10），不足则全部保留并给出提示
- 可生成多个 episode（不同随机子集）以评估稳定性（--episodes）
- 保留输入 CSV 的所有列，保证与现有 Dataset 兼容（例如：image_name, label_61, crop_id, disease_id, severity）
- 只筛子集，不改动文件结构，不破坏“用户空间”

用法：
    python create_fewshot_subset.py \
        --input data/cleaned/metadata/train_metadata.csv \
        --output data/cleaned/metadata/train_metadata_fewshot_10.csv \
        --per-class 10 \
        --seed 42

生成多个 episode（few-shot 训练常用做法）：
    python create_fewshot_subset.py \
        --input data/cleaned/metadata/train_metadata.csv \
        --output data/cleaned/metadata/train_metadata_fewshot_10.csv \
        --per-class 10 \
        --seed 42 \
        --episodes 5

输出：
- 若 episodes=1：直接输出到 --output 指定的路径
- 若 episodes>1：在 --output 路径的基础上追加后缀 _ep{1..E}.csv

后续训练建议（示例）：
- 使用我们的训练脚本：
  python train.py \
    --model-type fewshot \
    --backbone resnet50 \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
    --epochs 40 \
    --batch-size 16 \
    --lr 3e-4 \
    --fewshot-head-lr-scale 5.0 \
    --fewshot-freeze-backbone \
    --mixup-alpha 0.2 --cutmix-alpha 0.0 \
    --save-dir checkpoints/task2_fewshot

注意：
- 本脚本不会修改原始图像或目录结构。
- 若输入 metadata 中不存在某些列（如 is_duplicate），不会进行额外过滤。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _infer_class_column(df: pd.DataFrame) -> str:
    """
    推断类别列名。项目内使用 label_61 作为主类标签。
    """
    if "label_61" in df.columns:
        return "label_61"
    # 兜底：尝试常见列名
    for candidate in ["label", "class_id", "category_id"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Cannot infer class column. Expected 'label_61' (preferred) or one of ['label', 'class_id', 'category_id']."
    )


def _drop_duplicates_if_flagged(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果存在 is_duplicate 列（bool/int），则删除被标记的重复样本。
    否则原样返回（不做“猜测性”过滤，避免误删）。
    """
    if "is_duplicate" in df.columns:
        before = len(df)
        if df["is_duplicate"].dtype == bool:
            df = df[~df["is_duplicate"]].copy()
        else:
            # 约定：1/True 表示重复
            df = df[df["is_duplicate"] == 0].copy()
        after = len(df)
        if after < before:
            print(f"[Info] Dropped {before - after} rows flagged as duplicates via 'is_duplicate'.")
    return df


def _sample_per_class(
    df: pd.DataFrame,
    class_col: str,
    per_class: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    核心采样逻辑：每个类别随机抽取 per_class 样本；若不足 per_class，则全部保留。

    返回：
    - 采样后的子集 DataFrame（乱序）
    - 稀有类列表 [(class_id, count), ...]（样本数不足 per_class）
    """
    rnd = random.Random(seed)
    # 为保证可复现，用 pandas.sample(random_state=...) 且每类用不同子种子
    fewshot_parts = []
    rare_classes: List[Tuple[int, int]] = []

    # 分组并按类处理
    grouped = df.groupby(class_col, sort=True)
    class_ids = list(grouped.groups.keys())
    class_ids_sorted = sorted(class_ids)

    for idx, cls in enumerate(class_ids_sorted):
        group = grouped.get_group(cls)
        n = len(group)
        if n >= per_class:
            # 使用不同子种子以确保每类随机不同，但可复现
            sub_seed = (seed * 10007 + idx * 7919) % (2**32 - 1)
            sampled = group.sample(n=per_class, random_state=sub_seed)
            fewshot_parts.append(sampled)
        else:
            fewshot_parts.append(group)
            rare_classes.append((cls, n))

    out_df = pd.concat(fewshot_parts, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out_df, rare_classes


def create_fewshot_subset(
    input_meta_path: str,
    output_meta_path: str,
    per_class: int = 10,
    seed: int = 42,
    episodes: int = 1,
) -> List[Path]:
    """
    从输入 metadata 生成 N-shot 子集（可多 episode）。

    Args:
        input_meta_path: 输入 metadata CSV
        output_meta_path: 输出 metadata CSV（episodes>1 时追加 _ep{1..E} 后缀）
        per_class: 每类样本数（默认 10）
        seed: 随机种子
        episodes: 生成多少个不同 episode（默认 1）

    Returns:
        生成的所有输出文件路径列表
    """
    input_path = Path(input_meta_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_meta_path}")

    df = pd.read_csv(input_meta_path)
    print("=" * 60)
    print("Create Few-Shot Metadata (Task 2)")
    print("=" * 60)
    print(f"Input: {input_meta_path}")
    print(f"Total samples: {len(df):,}")

    class_col = _infer_class_column(df)
    print(f"Class column: {class_col}")

    # 可选去重（仅当存在 is_duplicate 列）
    df = _drop_duplicates_if_flagged(df)

    # 类别数量检查
    n_classes = df[class_col].nunique()
    print(f"Detected {n_classes} classes in metadata.")
    if n_classes < 61:
        print(f"[Warn] Less than 61 classes found ({n_classes}). Check your metadata.")
    elif n_classes > 61:
        print(f"[Info] More than 61 classes found ({n_classes}). Proceeding with all classes.")

    outputs: List[Path] = []
    base_output = Path(output_meta_path)
    base_output.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        ep_seed = (seed + ep * 13331) % (2**32 - 1)
        fewshot_df, rare = _sample_per_class(
            df, class_col=class_col, per_class=per_class, seed=ep_seed
        )

        # 保存
        if episodes == 1:
            out_path = base_output
        else:
            stem = base_output.stem
            suffix = base_output.suffix or ".csv"
            out_path = base_output.with_name(f"{stem}_ep{ep + 1}{suffix}")

        fewshot_df.to_csv(out_path, index=False)
        outputs.append(out_path)

        # 统计信息
        counts = fewshot_df[class_col].value_counts().sort_index()
        kept_min = int(counts.min())
        kept_max = int(counts.max())
        kept_mean = float(counts.mean())
        print("\n----------------------------------------")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"  Few-shot per class target: {per_class}")
        print(f"  Output samples: {len(fewshot_df):,}")
        print(f"  Per-class kept: min={kept_min}, max={kept_max}, mean={kept_mean:.2f}")
        if rare:
            print(f"  Rare classes (<{per_class} samples): {len(rare)}")
            preview = ", ".join([f"{cid}:{cnt}" for cid, cnt in rare[:10]])
            if len(rare) > 10:
                preview += ", ..."
            print(f"    {preview}")
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("✅ Few-shot metadata created successfully")
    print("=" * 60)
    print("Next steps:")
    if episodes == 1:
        print(f"  - Use: --train-meta {outputs[0]}")
    else:
        print(f"  - Choose one episode to train, e.g.: --train-meta {outputs[0]}")
    print("  - Keep validation metadata unchanged for evaluation.")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create N-shot per class metadata for Task 2")
    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned/metadata/train_metadata.csv",
        help="Input metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cleaned/metadata/train_metadata_fewshot_10.csv",
        help="Output few-shot metadata CSV",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="Number of samples per class (N-shot)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of different few-shot episodes to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_fewshot_subset(
        input_meta_path=args.input,
        output_meta_path=args.output,
        per_class=args.per_class,
        seed=args.seed,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
