import os

# 提交信息路径
COMMIT_INFO_PATH = r"D:\xz\科研\commit message generation\ATOM\data\ast_process\data\raw_commits_from_github"
# 提取函数信息路径
FUNCTIONS_EXTRACTED_PATH = r"D:\xz\科研\commit message generation\ATOM\data\ast_process\data\functions_extracted_commits"

# 保存数据集路径
DATASET_PATH = './datasets'
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)