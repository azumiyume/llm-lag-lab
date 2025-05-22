# ▶️ db_utils.py
# - DB接続・INSERT・検索（pgvector使用）
#main_ingest & main_queryで使用

import psycopg2
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv

load_dotenv()
CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")

def save_to_db(content, embedding, pdf_name):
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (content, embedding, pdf_name) VALUES (%s, %s, %s);",
            (content, embedding, pdf_name)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB保存エラー: {e}")

def search_similar(embedding, top_k=3):
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute("SELECT content, embedding, pdf_name FROM documents;")
        rows = cur.fetchall()
        results = []
        for content, emb, pdf_name in rows:
            emb_tensor = torch.tensor(eval(emb), device="cpu")
            sim = F.cosine_similarity(torch.tensor(embedding).unsqueeze(0), emb_tensor.unsqueeze(0)).item()
            results.append((content, sim, pdf_name))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    except Exception as e:
        print(f"❌ DB検索エラー: {e}")
        return []

def get_metadata_by_pdf_name(pdf_name):
    import psycopg2
    import os
    from dotenv import load_dotenv

    load_dotenv()
    conn = psycopg2.connect(os.getenv("PGVECTOR_CONNECTION_STRING"))
    cur = conn.cursor()
    cur.execute("""
        SELECT title, author, affiliation, publication_date
        FROM metadata
        WHERE pdf_name = %s
        LIMIT 1
    """, (pdf_name,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return {
            "title": result[0],
            "author": result[1],
            "affiliation": result[2],
            "publication_date": result[3],
        }
    return {}
