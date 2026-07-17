from src.text_preprocessing import VocabularyTokenizer,clean_text,tokenize

def test_clean_text_preserves_signal_tokens():
    cleaned=clean_text(
        "WIN £500 now! Call +1 303 555 0199 or visit https://example.com"
    )
    assert "<currency>" in cleaned
    assert "<number>" in cleaned
    assert "<phone>" in cleaned
    assert "<url>" in cleaned
    assert "!" in cleaned

def test_tokenizer_uses_pad_and_oov():
    tokenizer=VocabularyTokenizer(20).fit(["free prize now","meeting at home"])
    assert tokenizer.word_index["<pad>"]==0
    assert tokenizer.word_index["<oov>"]==1
    assert tokenizer.encode("unknownterm")[0]==1
    assert "<url>" in tokenize("visit https://example.com")
