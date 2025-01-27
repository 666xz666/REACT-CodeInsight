import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    RobertaModel,
    PLBartTokenizer,
    PLBartModel,
)
from config import (
    AST_TOKENIZER_MODEL,
    DIFF_TOKENIZER_MODEL,
    MODEL_PATH,
    DIFF_MODEL_TYPE,
    AST_MODEL_TYPE,
    TOKENIZER_PATH
)
from torch_multi_head_attention import MultiHeadAttention
import logging


def get_tokenizer(model_name, model_type):
    tokenizer = None
    tokenizer_model_path = str(os.path.join(TOKENIZER_PATH, model_name+'_react_tokenizer'))
    if model_type == "t5":
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)
    elif model_type == "plbart":
        tokenizer = PLBartTokenizer.from_pretrained(tokenizer_model_path, src_lang="java",
                                                    tgt_lang="en_XX", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)

    return tokenizer


def get_encoder(model_name, model_type):
    encoder = None
    model_path = str(os.path.join(MODEL_PATH, model_name))
    if model_type == "t5":
        encoder = RobertaModel.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == "plbart":
        encoder = PLBartModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    return encoder


class DualEncoder(nn.Module):
    def __init__(
            self, diff_model_name=DIFF_TOKENIZER_MODEL,
            ast_model_name=AST_TOKENIZER_MODEL,
            diff_model_type=DIFF_MODEL_TYPE,
            ast_model_type=AST_MODEL_TYPE,
            max_length=512,
    ):
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(DualEncoder, self).__init__()

        logging.info("DualEncoder model initializing...")
        # # 初始化diff编码器
        self.diff_encoder = get_encoder(diff_model_name, diff_model_type)
        self.diff_tokenizer = get_tokenizer(diff_model_name, diff_model_type)
        # # 初始化AST编码器
        self.ast_encoder = get_encoder(ast_model_name, ast_model_type)
        self.ast_tokenizer = get_tokenizer(ast_model_name, ast_model_type)
        logging.info("DualEncoder model initialized.")

        # 多模态融合层
        self.mlp = nn.Sequential(
            nn.Linear(self.diff_encoder.config.hidden_size * 2, self.diff_encoder.config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.diff_encoder.config.hidden_size * 4, self.diff_encoder.config.hidden_size)
        )

        # 自注意力层用于特征选择和降维
        self.self_attention = MultiHeadAttention(self.diff_encoder.config.hidden_size, head_num=8)

        # 将模型移动到指定的设备上
        if self.device is not None:
            self.diff_encoder.to(self.device)
            self.ast_encoder.to(self.device)
            self.mlp.to(self.device)
            self.self_attention.to(self.device)

    def encode_diff(self, diff_text):
        # 对diff文本进行编码
        diff_inputs = self.diff_tokenizer(
            diff_text, return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.max_length,
        )
        diff_inputs = {k: v.to(self.device) for k, v in diff_inputs.items()}
        diff_outputs = self.diff_encoder(**diff_inputs)
        return diff_outputs.last_hidden_state

    def encode_ast(self, ast_edits):
        # 对AST编辑序列进行编码
        ast_inputs = self.ast_tokenizer(
            ast_edits, return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.max_length,
        )
        ast_inputs = {k: v.to(self.device) for k, v in ast_inputs.items()}
        ast_outputs = self.ast_encoder(**ast_inputs)
        return ast_outputs.last_hidden_state

    def forward(self, diff_text, ast_edits):
        # 分别获取diff和AST的编码
        diff_encoded = self.encode_diff(diff_text)
        ast_encoded = self.encode_ast(ast_edits)

        # 多模态融合
        # 确保序列长度一致
        seq_length = min(diff_encoded.size(1), ast_encoded.size(1))
        diff_encoded = diff_encoded[:, :seq_length]
        ast_encoded = ast_encoded[:, :seq_length]
        combined = torch.cat((diff_encoded, ast_encoded), dim=-1)  # 假设batch维度为0

        # 使用MLP进行融合
        fused_output_mlp = self.mlp(combined)

        # 使用自注意力层进行特征选择和降维
        attn_output = self.self_attention(fused_output_mlp, fused_output_mlp, fused_output_mlp)

        return attn_output


if __name__ == '__main__':
    from config import AST_TOKENIZER_MODEL, DIFF_TOKENIZER_MODEL, MODEL_PATH
    import os

    # 使用示例
    dual_encoder = DualEncoder()

    # 假设的diff文本和AST编辑序列
    diff_text = ("diff --git a/file.py b/file.py\nindex 1234..5678 100644\n--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,"
                 "3 @@\n print('Hello')\n-print('World')\n+print('Transformer')")
    ast_edits = ("MethodDeclaration LocalVariableDeclaration StatementExpression +LocalVariableDeclaration "
                 "StatementExpression TryStatement ReturnStatement")

    # 获取编码
    diff_encoded = dual_encoder.encode_diff(diff_text)
    ast_encoded = dual_encoder.encode_ast(ast_edits)

    # 多模态融合
    fused_output = dual_encoder(diff_text, ast_edits)

    # 输出结果
    print(fused_output.shape)
    print(fused_output)
