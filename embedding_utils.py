# ▶️ embedding_utils.py
# - PLaMo埋め込みモデルのロードとベクトル生成
#main_ingest & main_queryで使用

import torch
from transformers import AutoTokenizer, AutoModel

class PlamoEmbedder:
    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True).to(device)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            return self.model.encode_document(text, self.tokenizer).squeeze().to("cpu").tolist()

