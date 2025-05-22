# ▶️ main_query.py
# - ユーザー入力を受けて検索・RakutenAI応答を生成する実行ファイル

# ▶️ main_ingest.py
# - PDFを読み込んでチャンク分割・埋め込み生成・DB保存まで行う実行ファイル

# ▶️ pdf_utils.py
# - PDF読み取りとチャンク化（text化・セクション分割）

# ▶️ embedding_utils.py
# - PLaMo埋め込みモデルのロードとベクトル生成

# ▶️ db_utils.py
# - DB接続・INSERT・検索（pgvector使用）

# ▶️ response_generator.py
# - RakutenAIモデルのロードと応答生成


from embedding_utils import PlamoEmbedder
from db_utils import search_similar
from response_generator import RakutenResponder

def main():
    embedder = PlamoEmbedder()
    responder = RakutenResponder()

    print("質問を入力してください（q で終了）")
    while True:
        query = input("ユーザー > ")
        if query.strip().lower() == "q":
            break
        if not query.strip():
            print("⚠️ 入力が空です。もう一度入力してください。")
            continue

        query_embedding = embedder.encode(query)
        results = search_similar(query_embedding)

        if results:
            print("✅ 関連チャンク:")
            for i, (content, score) in enumerate(results, start=1):
                print(f"{i}. {content[:100]}...（類似度: {score:.2f}）\n")
            context = "\n".join([content for content, _ in results])
        else:
            print("❌ 関連情報が見つかりませんでした。")
            context = ""

        response = responder.generate(context, query)
        print(f"アシスタント > {response}\n")

if __name__ == "__main__":
    main()
