# - PDFを読み込んでチャンク分割・埋め込み生成・DB保存まで行う実行ファイル

from pdf_utils import pdf_to_markdown_chunks
from embedding_utils import PlamoEmbedder
from db_utils import save_to_db
import os

def main():
    pdf_path = "/home/ts-lab/Master-summary/16_0006.pdf"
    pdf_name = os.path.basename(pdf_path)  # ← "16_0006.pdf"

    print("✅ PDFをMarkdown形式に整形中（GPT使用）...")
    chunks = pdf_to_markdown_chunks(pdf_path)

    if not chunks:
        print("❌ チャンクが空です。終了します。")
        return

    print(f"✅ 作成したチャンク数: {len(chunks)}")

    embedder = PlamoEmbedder()

    for chunk in chunks:
        embedding = embedder.encode(chunk)
        save_to_db(chunk, embedding, pdf_name) 
        print("✅ 保存完了:", chunk[:30], "...")

if __name__ == "__main__":
    main()
