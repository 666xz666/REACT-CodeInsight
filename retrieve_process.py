import os
import pandas as pd
from config import DATASET_PATH, MAX_LEN
# from react_code_insight import DualEncoder
from react_code_insight import get_embedding_batch
import logging
import torch
import numpy as np
from tqdm import tqdm


def merge_csv_files(input_folder, output_file):
    """
    合并指定文件夹中的所有 CSV 文件，并将结果保存到一个新的 CSV 文件中。
    """
    # 获取文件夹中所有 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # 检查是否有 CSV 文件
    if not csv_files:
        print("未找到任何 CSV 文件。")
        return

    # 打开输出文件
    with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
        # 遍历每个 CSV 文件
        for file in csv_files:
            file_path = os.path.join(input_folder, file)
            # 分块读取文件
            chunk_size = 10000  # 每次读取的行数
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # 将分块数据写入输出文件
                chunk.to_csv(output_csv, index=False, header=(file == csv_files[0]), mode='a')
                print(f"已处理文件 {file}")

    print(f"所有 CSV 文件已成功合并并保存到 {output_file}")


def encode(file_path, output_folder, batch_size=32):
    """
    将编码结果添加到 CSV 文件
    :param file_path: CSV 文件路径
    :param output_folder: 输出文件夹路径
    :param batch_size: 每个批次的大小
    :return:
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保 CSV 文件中包含所需的列
    if 'diff' not in df.columns or 'ast_seq' not in df.columns:
        raise ValueError(f"CSV 文件 {file_path} 必须包含 'diff' 和 'ast_seq' 列")

    # 初始化一个空列表来存储特征向量
    feature_vectors = []

    # 获取数据的总行数
    total_rows = len(df)

    # 分批处理数据
    for start_idx in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_rows)

        # 提取当前批次的 diff 和 ast_seq 数据
        batch_diff_texts = df['diff'][start_idx:end_idx].tolist()
        batch_ast_sequences = df['ast_seq'][start_idx:end_idx].tolist()

        # 替换 NaN 值为默认值（空字符串）
        batch_ast_sequences = [seq if not isinstance(seq, float) or not np.isnan(seq) else "" for seq in batch_ast_sequences]

        # 调用批量嵌入函数
        with torch.no_grad():  # 不计算梯度，节省内存
            batch_fused_outputs = get_embedding_batch(batch_diff_texts, batch_ast_sequences)

        # 将生成的特征向量存储到列表中
        feature_vectors.extend(batch_fused_outputs)

    # 将特征向量添加到 DataFrame 中
    df['feature_vector'] = feature_vectors

    # 生成输出文件路径
    output_file = os.path.join(output_folder, os.path.basename(file_path))

    # 保存更新后的 DataFrame 到 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"特征向量已成功添加到 {output_file}")

def main():
    # input_folder = DATASET_PATH  # 替换为你的 CSV 文件所在文件夹路径
    # output_folder = os.path.join(DATASET_PATH, 'encoder_datasets_1')  # 替换为输出文件夹路径
    #
    # # 确保输出文件夹存在
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #
    # #获取文件夹中所有 CSV 文件
    # csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    #
    # # 检查是否有 CSV 文件
    # if not csv_files:
    #     print("未找到任何 CSV 文件。")
    #     return
    #
    # # 逐个处理每个 CSV 文件
    # for file in csv_files:
    #     file_path = os.path.join(input_folder, file)
    #     output_file = os.path.join(output_folder, file)
    #
    #     # 检查输出文件是否已经存在
    #     if os.path.exists(output_file):
    #         print(f"输出文件 {output_file} 已经存在，跳过处理。")
    #         continue
    #
    #     # 调用 encode 函数处理文件
    #     encode(file_path, output_folder)

    merge_csv_files("./datasets/encoder_datasets_1", "./datasets/merged/source.csv")


if __name__ == '__main__':
    main()
