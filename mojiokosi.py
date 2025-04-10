import speech_recognition as sr
import MeCab
import csv

# --- カスタム辞書の読み込み ---
class CustomDictionary:
    def __init__(self):
        self.entries = {}

    def load_from_csv(self, filename="custom_dictionary.csv"):
        with open(filename, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word = row["表層形"]
                self.entries[word] = {
                    "品詞": row["品詞"],
                    "難易度スコア": int(row["難易度スコア"])
                }

    def get_difficulty(self, word):
        return self.entries.get(word, {"難易度スコア": "不明"})["難易度スコア"]

# --- MeCab解析 ---
def analyze_text(text, dictionary):
    mecab = MeCab.Tagger()
    result = []
    node = mecab.parseToNode(text)

    while node:
        word = node.surface
        pos = node.feature.split(",")[0]  # 品詞情報
        if word:  # 空文字チェック
            difficulty = dictionary.get_difficulty(word)
            result.append(f"{word} ({pos}) - 難易度: {difficulty}")
        node = node.next
    return result

# --- 音声入力 (文字起こし) ---
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ 話してください...")
        audio = recognizer.listen(source)

    try:
        print("📝 テキスト変換中...")
        text = recognizer.recognize_google(audio, language="ja-JP")
        print(f"🗣️ 音声認識結果: {text}")
        return text
    except sr.UnknownValueError:
        print("❗ 音声が認識できませんでした")
        return ""
    except sr.RequestError:
        print("❗ Google Speech Recognition サービスに接続できませんでした")
        return ""

# --- メイン処理 ---
if __name__ == "__main__":
    dictionary = CustomDictionary()
    dictionary.load_from_csv("custom_dictionary.csv")  # 辞書ファイルの読み込み

    text = recognize_speech()  # 音声入力
    if text:
        analysis = analyze_text(text, dictionary)
        print("\n🔎 MeCab解析結果:")
        for result in analysis:
            print(result)
