from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./RakutenAI-2.0-8x7B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("チャット開始！（q で終了）")

while True:
    user_input = input("ユーザー > ")
    if user_input.strip().lower() == "q":
        break

    # 会話形式プロンプトに統一（"RakutenAI" 表示なし）
    prompt = f"""以下はユーザーとアシスタントの会話です。
ユーザー: {user_input}
アシスタント:"""

    # 入力をトークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()

    max_new_tokens = 256  # 🔺ここを大きくすることで途中で切れにくくなる

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # レスポンス部分のみ抽出
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = decoded.split("アシスタント:")[-1].strip()

    print("アシスタント >", response)
