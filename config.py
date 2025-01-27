import os

# 提交信息路径
COMMIT_INFO_PATH = r"D:\xz\科研\commit message generation\ATOM\data\ast_process\data\raw_commits_from_github"
# 提取函数信息路径
FUNCTIONS_EXTRACTED_PATH = r"D:\xz\科研\commit message generation\ATOM\data\ast_process\data\functions_extracted_commits"

# 预训练模型路径
MODEL_PATH = r"D:\xz\科研\commit message generation\RAG\REACT-replication\data\checkpoints"
MODEL_LIST = ['codet5', 'codet5p', 'plbart', 'unixcoder']

AST_TOKENIZER_MODEL = 'codet5'
DIFF_TOKENIZER_MODEL = 'plbart'

TOKENIZER_PATH = r'D:\xz\科研\commit message generation\RAG\REACT-replication\data\tokenizers'

AST_MODEL_TYPE = 't5'
DIFF_MODEL_TYPE = 'plbart'

# 保存数据集路径
DATASET_PATH = './datasets'
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
