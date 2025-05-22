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
from db_utils import search_similar, get_metadata_by_pdf_name

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
            for i, (content, score, _) in enumerate(results, start=1):  # pdf_name を含む3要素タプル
                print(f"{i}. {content[:100]}...（類似度: {score:.2f}）\n")

            # ✅ pdf_name を取得（最も類似したチャンクから）
            top_pdf_name = results[0][2]
            metadata = get_metadata_by_pdf_name(top_pdf_name)
            meta_info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            
            print("メタデータ:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")

            # ✅ context の作成（チャンクまとめ）
            context = "\n".join([content for content, _, _ in results])

            # ✅ 応答生成（meta_info を追加）
            response = responder.generate(context, query, meta_info)



        else:
            print("❌ 関連情報が見つかりませんでした。")
            response = responder.generate("", query)

        print(f"アシスタント > {response}\n")

if __name__ == "__main__":
    main()
