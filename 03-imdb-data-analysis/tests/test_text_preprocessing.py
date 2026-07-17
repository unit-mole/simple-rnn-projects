from src.text_preprocessing import VocabularyTokenizer, clean_text, tokenize


def test_clean_text_removes_html_and_control_tokens():
    cleaned = clean_text("START Great<br />movie UNK!!!")
    assert "start" not in cleaned
    assert "<br" not in cleaned
    assert "<unk>" in cleaned
    assert "great" in cleaned


def test_tokenizer_has_pad_and_oov():
    tokenizer = VocabularyTokenizer(max_words=8).fit(
        ["good movie", "bad movie", "not good"]
    )
    assert tokenizer.word_index["<pad>"] == 0
    assert tokenizer.word_index["<oov>"] == 1
    encoded = tokenizer.encode("unknownword movie")
    assert encoded[0] == 1
    assert len(tokenize("not good!")) == 3
