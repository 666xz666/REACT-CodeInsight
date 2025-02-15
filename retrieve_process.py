import os
import pandas as pd
from config import DATASET_PATH, MAX_LEN
from react_code_insight import get_embedding_batch, EnhancedRetriever
import logging
import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        batch_ast_sequences = [seq if not isinstance(seq, float) or not np.isnan(seq) else "" for seq in
                               batch_ast_sequences]

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


def split_csv_data(file_path, train_size, valid_size, test_size, output_dir=".", random_state=42):
    """
    将 CSV 文件划分为训练集、验证集、测试集和剩余数据集。
    支持分块读取和分块保存以优化内存使用。

    参数:
        file_path (str): 输入的 CSV 文件路径。
        train_size (int): 训练集的大小。
        valid_size (int): 验证集的大小。
        test_size (int): 测试集的大小。
        output_dir (str): 输出文件夹路径，默认为当前目录。
        random_state (int): 随机种子，确保结果可复现。
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化分块大小
    chunk_size = 10000  # 每块读取的行数，可以根据需要调整
    total_size = train_size + valid_size + test_size

    # 初始化数据集
    train_data = []
    valid_data = []
    test_data = []
    rag_data = []

    # 分块读取数据
    try:
        with tqdm(total=os.path.getsize(file_path), desc="读取 CSV 文件", unit="B", unit_scale=True) as pbar:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk = chunk.sample(frac=1, random_state=random_state).reset_index(drop=True)  # 打乱当前块
                for row in chunk.itertuples(index=False):
                    if len(train_data) < train_size:
                        train_data.append(row)
                    elif len(valid_data) < valid_size:
                        valid_data.append(row)
                    elif len(test_data) < test_size:
                        test_data.append(row)
                    else:
                        rag_data.append(row)
                pbar.update(chunk.memory_usage(deep=True).sum())
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 检查数据量是否足够
    if len(train_data) < train_size or len(valid_data) < valid_size or len(test_data) < test_size:
        print("错误：数据量不足以满足划分需求。")
        return

    # 分块保存数据集
    def save_data(data, file_name):
        chunk_size = 10000  # 每块保存的行数
        with open(os.path.join(output_dir, file_name), "w", newline="", encoding="utf-8") as f:
            header = True
            for i in tqdm(range(0, len(data), chunk_size), desc=f"保存 {file_name}"):
                chunk = pd.DataFrame(data[i:i + chunk_size])
                chunk.to_csv(f, header=header, index=False, mode="a")
                header = False

    # 保存数据集
    save_data(train_data, "train.csv")
    save_data(valid_data, "valid.csv")
    save_data(test_data, "test.csv")
    save_data(rag_data, "rag_knowledge_base.csv")

    print(f"数据划分完成！文件已保存到 {output_dir} 目录。")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(valid_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"RAG 知识库大小: {len(rag_data)}")



def main():
    input_folder = DATASET_PATH  # 替换为你的 CSV 文件所在文件夹路径
    output_folder = os.path.join(DATASET_PATH, 'encoder_datasets_1')  # 替换为输出文件夹路径

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logging.info(f"输入文件夹: {input_folder}")
    logging.info(f"输出文件夹: {output_folder}")

    logging.info("开始编码数据...")
    # 获取文件夹中所有 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # 检查是否有 CSV 文件
    if not csv_files:
        print("未找到任何 CSV 文件。")
        return

    # 逐个处理每个 CSV 文件
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, file)

        # 检查输出文件是否已经存在
        if os.path.exists(output_file):
            print(f"输出文件 {output_file} 已经存在，跳过处理。")
            continue

        # 调用 encode 函数处理文件
        encode(file_path, output_folder)

    logging.info("编码数据完成！")

    merged_path = os.path.join(DATASET_PATH, 'merged/source.csv')

    # 合并数据
    if os.path.exists(merged_path):
        logging.warning(f"输出文件 {merged_path} 已经存在，跳过处理。")
    else:
        logging.info(f"开始合并数据...")
        merge_csv_files(output_folder, merged_path)
        logging.info(f"合并数据完成！")

    splitted_path = os.path.join(DATASET_PATH, 'splitted_data')
    logging.info(f"开始划分数据集...")
    if os.path.exists(splitted_path):
        logging.warning(f"输出目录 {splitted_path} 已经存在，跳过处理。")
    else:
        # 划分数据集
        split_csv_data(
            file_path=merged_path,  # 输入文件路径
            train_size=75000,  # 训练集大小
            valid_size=7500,  # 验证集大小
            test_size=7500,  # 测试集大小
            output_dir=splitted_path,  # 输出目录
            random_state=114514  # 随机种子
        )


if __name__ == '__main__':
    main()
