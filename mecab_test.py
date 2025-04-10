import MeCab

mecab = MeCab.Tagger()
text = "Sudashiは辞書ベースの高精度な解析を行うが、処理時間が長いってこと？"
result = mecab.parse(text)

print("MeCab解析結果:\n", result)
