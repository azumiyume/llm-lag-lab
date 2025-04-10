import csv
import re

# カスタム辞書のエントリーフォーマット
class DictionaryEntry:
    def __init__(self, word, pos, difficulty):
        self.word = word            # 語彙
        self.pos = pos              # 品詞 (e.g., 名詞, 動詞)
        self.difficulty = difficulty  # 難易度スコア (例: 1〜5段階)

    def to_csv_row(self):
        # MeCab形式 (表層形, 品詞, 読み, 説明)
        return [self.word, self.pos, "*", "*", "*", "*", "*", self.difficulty]

# 辞書データ管理クラス
class CustomDictionary:
    def __init__(self):
        self.entries = []

    def add_entry(self, word, pos, difficulty):
        entry = DictionaryEntry(word, pos, difficulty)
        self.entries.append(entry)

    def save_to_csv(self, filename="custom_dictionary.csv"):
        with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["表層形", "品詞", "読み", "*", "*", "*", "*", "難易度スコア"])
            for entry in self.entries:
                writer.writerow(entry.to_csv_row())
        print(f"{filename} に保存しました。")

    def load_from_csv(self, filename="custom_dictionary.csv"):
        with open(filename, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.add_entry(row["表層形"], row["品詞"], row["難易度スコア"])

    def search_word(self, query):
        pattern = re.compile(query)
        results = [entry for entry in self.entries if pattern.search(entry.word)]
        return results

# --- サンプル実装 ---
if __name__ == "__main__":
    dictionary = CustomDictionary()

    # サンプルデータの追加
    #1:小学生　2：中高生　3:大学生　4：理解が難しい抽象的　5:難解
    # 理解しやすい語彙
    dictionary.add_entry("簡単", "形容詞", "1")
    dictionary.add_entry("基本", "名詞", "2")
    dictionary.add_entry("わかりやすい", "形容詞", "1")

    # 理解が難しい語彙
    dictionary.add_entry("難解", "形容詞", "3")
    dictionary.add_entry("複雑", "形容詞", "2")
    dictionary.add_entry("冗長", "名詞", "4")

    # 抽象的な語彙
    dictionary.add_entry("概念", "名詞", "3")
    dictionary.add_entry("認識", "名詞", "3")
    dictionary.add_entry("仮説", "名詞", "4")

    # 専門用語
    dictionary.add_entry("ニューラルネットワーク", "名詞", "5")
    dictionary.add_entry("エントロピー", "名詞", "5")

    #色々
    dictionary.add_entry("わかる", "動詞", "1")
    dictionary.add_entry("できる", "動詞", "1")
    dictionary.add_entry("もう一回", "副詞", "2")
    dictionary.add_entry("みたいな", "助詞", "2")
    dictionary.add_entry("簡単に言うと", "副詞", "2")
    dictionary.add_entry("要するに", "接続詞" ,"3")
    dictionary.add_entry("逆に", "接続詞", "3")
    dictionary.add_entry("わからない", "形容詞", "1")
    dictionary.add_entry("なんとなく", "副詞", "2")
    dictionary.add_entry("だいたい", "副詞", "3")
    dictionary.add_entry("とりあえず", "副詞", "3")
    dictionary.add_entry("いまいち", "副詞", "3")
    dictionary.add_entry("難しい", "形容詞", "2")
    dictionary.add_entry("どういうこと", "疑問詞", "3")
    dictionary.add_entry("なるほど", "感動詞","4")
    dictionary.add_entry("確かに", "副詞", "4")
    dictionary.add_entry("明確に", "副詞", "4")
    dictionary.add_entry("根拠として", "接続詞", "5")
    dictionary.add_entry("裏付ける", "動詞", "5")
    dictionary.add_entry("結論として", "接続詞", "5")
    dictionary.add_entry("なんか", "副詞", "2")
    dictionary.add_entry("うーん", "感動詞", "1")

    # CSVに保存
    dictionary.save_to_csv()

    # 語彙検索テスト
    print("\n[検索結果] '難'を含む単語:")
    results = dictionary.search_word("難")
    for result in results:
        print(f"{result.word} (難易度: {result.difficulty})")

