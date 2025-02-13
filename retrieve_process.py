import os
import pandas as pd
from config import DATASET_PATH, MAX_LEN
from react_code_insight import DualEncoder
import logging
import torch
import numpy as np

dual_encoder = None


def load_encoder():
    logging.info("Loading dual encoder...")
    global dual_encoder
    dual_encoder = DualEncoder()
    logging.info("Dual encoder loaded.")


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


def encode(file_path, output_folder):
    """
    将编码结果添加到csv文件
    :param file_path: CSV 文件路径
    :param output_folder: 输出文件夹路径
    :return:
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保 CSV 文件中包含所需的列
    if 'diff' not in df.columns or 'ast_seq' not in df.columns:
        raise ValueError(f"CSV 文件 {file_path} 必须包含 'diff' 和 'ast_seq' 列")

    # 初始化一个空列表来存储特征向量
    feature_vectors = []

    # 遍历每一行，进行编码和融合
    for index, row in df.iterrows():
        diff_text = row['diff']
        ast_edits = row['ast_seq']
        if isinstance(ast_edits, float) and np.isnan(ast_edits):
            ast_edits = [""]  # 替换 NaN 为默认值（空字符串）
        print("ast_edits:", ast_edits)
        print("Type of ast_edits:", type(ast_edits))
        # 使用 DualEncoder 进行编码和融合
        with torch.no_grad():  # 不计算梯度，节省内存
            diff_text = diff_text[:MAX_LEN]
            ast_edits = ast_edits[:MAX_LEN]
            fused_output = dual_encoder(diff_text, ast_edits)
        # 输出特征向量的形状和内容
        # print("Generated feature vector shape:", fused_output.shape)
        # print("Generated feature vector content:", fused_output)
        # 将特征向量转换为列表并存储
        feature_vectors.append(fused_output.cpu().numpy().tolist())

    # 将特征向量添加到 DataFrame 中
    df['feature_vector'] = feature_vectors

    # 生成输出文件路径
    output_file = os.path.join(output_folder, os.path.basename(file_path))

    # 保存更新后的 DataFrame 到 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"特征向量已成功添加到 {output_file}")


def main():
    # input_folder = DATASET_PATH  # 替换为你的 CSV 文件所在文件夹路径
    # output_folder = os.path.join(DATASET_PATH, 'encoder_datasets')  # 替换为输出文件夹路径
    #
    # # 确保输出文件夹存在
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #
    # # 加载编码器
    # load_encoder()
    #
    # # 获取文件夹中所有 CSV 文件
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

    source_path = "./datasets/source.csv"
    # merge_csv_files("./datasets/encoder_datasets", "./datasets/source.csv")
    
    


if __name__ == '__main__':
    main()
