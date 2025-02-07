import os
import pandas as pd
from config import DATASET_PATH
from react_code_insight import DualEncoder

dual_encoder = None

def load_encoder():
    global dual_encoder
    dual_encoder = DualEncoder()

def merge_csv_files(input_folder, output_file):
    """
    合并指定文件夹中的所有 CSV 文件，并将结果保存到一个新的 CSV 文件中。

    :param input_folder: 包含 CSV 文件的文件夹路径
    :param output_file: 合并后的 CSV 文件保存路径
    """
    # 获取文件夹中所有 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # 检查是否有 CSV 文件
    if not csv_files:
        print("未找到任何 CSV 文件。")
        return

    # 读取第一个 CSV 文件作为初始 DataFrame
    merged_df = pd.read_csv(os.path.join(input_folder, csv_files[0]))

    # 逐个读取并合并后续的 CSV 文件
    for file in csv_files[1:]:
        file_path = os.path.join(input_folder, file)
        temp_df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

    # 保存合并后的数据到新的 CSV 文件
    merged_df.to_csv(output_file, index=False)
    print(f"所有 CSV 文件已成功合并并保存到 {output_file}")


def encode(file_path):
    """
    将编码结果添加到csv文件
    :param file_path:
    :return:
    """
    pass

def main():
    input_folder = DATASET_PATH  # 替换为你的 CSV 文件所在文件夹路径
    merged_file = os.path.join(DATASET_PATH, 'merged_datasets', 'commits_ast.csv')  # 替换为合并后的文件保存路径

    # 合并 CSV 文件
    # merge_csv_files(input_folder, merged_file)

    #加载
    load_encoder()

    # 编码
    encode(merged_file)

if __name__ == '__main__':
    main()