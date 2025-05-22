# ▶️ response_generator.py
# - RakutenAIモデルのロードと応答生成
#main_queryで使用

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RakutenResponder:
    def __init__(self, model_path="./RakutenAI-2.0-8x7B-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    def generate(self, context, user_input, meta_info=""):
        prompt = (
            "あなたは親しみやすく丁寧に応答するアシスタントです。\n"
            "提供する情報は信頼性が高く、根拠が明確であるべきです。\n"
            "ここでの対話における研究などは、データベース内にある研究を指すものとします。\n"
            "回答は長すぎず短すぎない簡潔なものを目指してください。\n"
            "質問が曖昧な場合は、必要に応じて確認質問を挿入してください。\n"
            f"\nユーザーの質問: {user_input}"
            f"\n論文メタ情報:\n{meta_info}"
            f"\n関連情報:\n{context}\nアシスタント:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(256):
                outputs = self.model(input_ids=generated_ids)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).split("アシスタント:")[-1].strip()
