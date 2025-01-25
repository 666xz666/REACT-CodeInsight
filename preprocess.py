import os
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from config import COMMIT_INFO_PATH, FUNCTIONS_EXTRACTED_PATH, DATASET_PATH
from ast_tool import extract_ast_seq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_function_dict():
    """
    创建每个提交对应的函数文件名的字典，并保存到json文件中
    """
    dict = {}

    project_name_list = os.listdir(FUNCTIONS_EXTRACTED_PATH)
    for project_name in project_name_list:
        function_file_list = os.listdir(os.path.join(FUNCTIONS_EXTRACTED_PATH, project_name))

        for function_file in function_file_list:
            commit_id = function_file.split('.')[0].split('_')[1]
            dict[commit_id] = {
                'positive': None,
                'negative': None
            }

        for function_file in function_file_list:
            commit_id = function_file.split('.')[0].split('_')[1]
            type = function_file.split('.')[0].split('_')[2]
            dict[commit_id][type] = function_file

    with open(os.path.join(DATASET_PATH, 'dict.json'), 'w') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)

    logger.info('Function dictionary created at ' + os.path.join(DATASET_PATH, 'dict.json'))


def get_code_from_file(file_path):
    """
    从文件中读取代码
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    return code


def process_commit(row, project_name, dict):
    """
    处理单个commit的函数
    """
    try:
        commit_id = row['commit_id']
        diff = row['diff']
        message = row['message']

        neg_function_file = os.path.join(FUNCTIONS_EXTRACTED_PATH, project_name, dict[commit_id]['negative'])
        pos_function_file = os.path.join(FUNCTIONS_EXTRACTED_PATH, project_name, dict[commit_id]['positive'])

        neg_code = get_code_from_file(neg_function_file)
        pos_code = get_code_from_file(pos_function_file)

        ast_seq = extract_ast_seq(neg_code=neg_code, pos_code=pos_code)

        return {
            'commit_id': commit_id,
            'diff': diff,
            'ast_seq': ast_seq,
            'message': message
        }
    except Exception as e:
        # logger.error(f'Error in processing commit: {row["commit_id"]}')
        # logger.error(e)
        return None


def main():
    if not os.path.exists(os.path.join(DATASET_PATH, 'dict.json')):
        create_function_dict()
    else:
        logger.info('Function dictionary already exists at ' + os.path.join(DATASET_PATH, 'dict.json'))

    with open(os.path.join(DATASET_PATH, 'dict.json'), 'r') as f:
        dict = json.load(f)

    commit_csv_file_list = os.listdir(COMMIT_INFO_PATH)

    # 初始化一个空的DataFrame用于存储结果
    result_df = pd.DataFrame(columns=['commit_id', 'diff', 'ast_seq', 'message'])

    for commit_csv_file in commit_csv_file_list:
        project_name = commit_csv_file.split('.')[0].split('_')[0]

        commit_df = pd.read_csv(os.path.join(COMMIT_INFO_PATH, commit_csv_file),
                                usecols=['commit_id', 'diff', 'message'])

        # 使用多进程处理
        with Pool(processes=cpu_count()) as pool:
            # 使用partial函数固定部分参数
            partial_process_commit = partial(process_commit, project_name=project_name, dict=dict)
            results = list(tqdm(pool.imap(partial_process_commit, [row for _, row in commit_df.iterrows()]),
                                total=commit_df.shape[0], desc=f"Processing {project_name}"))

        # 将结果添加到DataFrame中
        for result in results:
            if result is not None:
                result_df = result_df.append(result, ignore_index=True)

    result_df.to_csv(os.path.join(DATASET_PATH, 'dataset.csv'), index=False)
    logger.info(f"Result saved to {os.path.join(DATASET_PATH, 'dataset.csv')}")


if __name__ == '__main__':
    main()