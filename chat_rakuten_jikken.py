from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import subprocess
import pandas as pd
from threading import Thread
from datetime import datetime

# ===========================
# 🔧 GPUモニタリング関数（表形式）
# ===========================
gpu_log = []
timestamps = []
monitoring_flag = False

def monitor_gpu_utilization(interval=1):
    global gpu_log, timestamps
    while monitoring_flag:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            encoding='utf-8'
        )
        usage = [int(line.strip()) for line in result.stdout.strip().split('\n')]
        gpu_log.append(usage)
        timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(interval)

def save_gpu_log_to_csv():
    if gpu_log:
        df = pd.DataFrame(gpu_log, columns=[f"GPU {i}" for i in range(len(gpu_log[0]))])
        df.insert(0, "Timestamp", timestamps)
        
        filename = f"gpu_log_wide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"📄 GPU使用率ログを保存しました（ワイド形式）: {filename}")

# ===========================
# 🔧 モデル準備
# ===========================
model_path = "./RakutenAI-2.0-8x7B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU情報出力
print("🟢 CUDA 利用可能 :", torch.cuda.is_available())
print(" 使用可能GPU数 :", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
print("\nチャット開始！（q で終了）")

# ===========================
# 🔧 対話ループ
# ===========================
while True:
    user_input = input("ユーザー > ")
    if user_input.strip().lower() == "q":
        break

    prompt = f"""以下はユーザーとアシスタントの会話です。\nユーザー: {user_input}\nアシスタント:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_ids = input_ids.clone()
    max_new_tokens = 256

    # 🔁 モニタリング開始
    gpu_log = []
    timestamps = []
    monitoring_flag = True
    monitor_thread = Thread(target=monitor_gpu_utilization)
    monitor_thread.start()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
    end_time = time.time()

    monitoring_flag = False
    monitor_thread.join()
    save_gpu_log_to_csv()

    # 応答出力
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).split("アシスタント:")[-1].strip()
    print("アシスタント >", response)
    print(f"🔹 応答生成時間: {end_time - start_time:.2f} 秒\n")
