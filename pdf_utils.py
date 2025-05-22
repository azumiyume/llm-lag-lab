import fitz
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# ✅ 環境変数の読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ PDF → テキスト
def read_pdf_text(pdf_path: str) -> str:
    all_text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

# ✅ 長文分割
def split_text_into_blocks(text, max_chars=2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# ✅ GPTでMarkdownに整形
def ask_gpt_markdown(text):
    system_prompt = """
[タスク]
ユーザーから入力された文章はPDFファイルから読み取られた文章です。
入力された文章をマークダウンに変換してください。
変換する際の注意点と出力形式は以下に示します。

[注意点]
- ユーザーから入力された文章の内容は改変せず、フォーマットのみを整えてください。
- PDFからの変換の影響で不要な改行やスペースが挿入されています。適宜削除してください。
- PDFからの変換の影響で表がテキストに変換されています。適宜表形式に変換してください。
- 図や画像は削除されています。画像のキャプションなどが残っている場合があります。これは`![図n キャプション名]()`のように変換してください。

[追加ルール]
- セクション（##）やサブセクション（###）が変わるたびに、その内容をひとつのチャンクになるように出力をまとめてください。
- ひとつのセクションやサブセクションの内容が複数段落にわたる場合は、必ずそれらをまとめてください。
- このチャンク分けは、後にプログラムでそのまま「見出し単位で分割」されることを前提に設計してください。

[出力形式]
- 出力の形式はMarkdown形式で出力してください。
- `実行が完了しました`等のMarkdown形式ではない部分は出力しないでください。
- 一番上位の```等のコードブロックは不要です。素のMarkdownのみ出力してください。

適切な実行を確保するために、出力は常にこの形式に従ってください。
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

# ✅ 分割しながらGPTでMarkdown化
def ask_gpt_markdown_in_chunks(full_text):
    blocks = split_text_into_blocks(full_text)
    markdown_parts = []
    for idx, block in enumerate(blocks):
        print(f"GPT処理中: ブロック{idx + 1}/{len(blocks)}")
        markdown = ask_gpt_markdown(block)
        markdown_parts.append(markdown)
    return "\n".join(markdown_parts)

# ✅ Markdown → チャンク
def split_into_chunks(markdown_text):
    chunks = re.split(r"(?=\n#{2,3} |\n\d+\.\d*)", markdown_text)
    return [c.strip() for c in chunks if c.strip()]

# ✅ パイプライン関数
def pdf_to_markdown_chunks(pdf_path: str):
    raw_text = read_pdf_text(pdf_path)
    markdown_text = ask_gpt_markdown_in_chunks(raw_text)
    return split_into_chunks(markdown_text)

# ✅ 単体実行
if __name__ == "__main__":
    pdf_path = "/home/ts-lab/Master-summary/16_0006.pdf"
    chunks = pdf_to_markdown_chunks(pdf_path)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- チャンク {i} ---\n{chunk}")
