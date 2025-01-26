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

def process_project(project_name, commit_csv_file, dict):
    """
    处理单个项目
    """
    try:
        commit_df = pd.read_csv(os.path.join(COMMIT_INFO_PATH, commit_csv_file),
                                usecols=['commit_id', 'diff', 'message'])

        # 使用partial函数固定部分参数
        partial_process_commit = partial(process_commit, project_name=project_name, dict=dict)
        results = list(tqdm(map(partial_process_commit, [row for _, row in commit_df.iterrows()]),
                            total=commit_df.shape[0], desc=f"Processing {project_name}"))


        result_file = os.path.join(DATASET_PATH, f"{project_name}_commits_ast.csv")
        # 将结果逐行写入文件
        with open(result_file, 'a', newline='', encoding='utf-8') as f:
            for result in results:
                if result is not None:
                    pd.DataFrame([result]).to_csv(f, header=f.tell() == 0, index=False)

        logger.info(f"Processed {project_name}")
        logger.info(f"Result saved to {result_file}")
        return project_name
    except Exception as e:
        # logger.error(f"Error processing project {project_name}: {e}")
        return None


def main():
    processed_commits = []
    if os.path.exists(os.path.join(DATASET_PATH, 'processed_commits.json')):
        logger.info('Read processed commits from file...')
        with open(os.path.join(DATASET_PATH, 'processed_commits.json'), 'r') as f:
            processed_commits = json.load(f)['processed_commits']
            logger.info(f"Processed commits: {processed_commits}")

    if not os.path.exists(os.path.join(DATASET_PATH, 'dict.json')):
        create_function_dict()
    else:
        logger.info('Function dictionary already exists at ' + os.path.join(DATASET_PATH, 'dict.json'))

    with open(os.path.join(DATASET_PATH, 'dict.json'), 'r') as f:
        dict = json.load(f)

    commit_csv_file_list = os.listdir(COMMIT_INFO_PATH)


    # 创建进程池
    pool = Pool(processes=cpu_count())

    try:
        # 提取项目名称列表
        project_names = [commit_csv_file.split('.')[0].split('_')[0] for commit_csv_file in commit_csv_file_list]
        # 去重并过滤已处理的项目
        project_names = list(set(project_names) - set(processed_commits))

        tmp = []
        for project_name in project_names:
            if os.path.exists(os.path.join(DATASET_PATH, f"{project_name}_commits_ast.csv")):
                processed_commits.append(project_name)
                logger.info(f"{project_name} already processed, skip...")
                continue
            else:
                tmp.append(project_name)
        project_names = tmp


        # 使用进程池并行处理多个项目
        partial_process_project = partial(process_project, dict=dict)
        project_args = [(project_name, f"{project_name}_commits.csv") for project_name in project_names]
        processed_projects = pool.starmap(partial_process_project, project_args)

        # 更新已处理的项目列表
        processed_commits.extend([project for project in processed_projects if project is not None])
        logger.info(f"Processed projects: {processed_commits}")
    except (KeyboardInterrupt, SystemExit) as e:
        logger.error(f"Program interrupted by user: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if pool:
            pool.close()  # 关闭进程池，不再接受新任务
            pool.join()  # 等待所有子进程完成
        # 保存已处理的项目
        with open(os.path.join(DATASET_PATH, 'processed_commits.json'), 'w') as f:
            json.dump({'processed_commits': processed_commits}, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()