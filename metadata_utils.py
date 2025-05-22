#ragdbのmetadataテーブル設定。ユーティリティモジュール
#test_insert_metadata.pyで登録,実行する
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

def save_metadata(pdf_name, title, author, affiliation, publication_date):
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO metadata (pdf_name, title, author, affiliation, publication_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (pdf_name) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                affiliation = EXCLUDED.affiliation,
                publication_date = EXCLUDED.publication_date;
        """, (pdf_name, title, author, affiliation, publication_date))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ 登録完了:", pdf_name)
    except Exception as e:
        print(f"❌ 登録エラー: {e}")

