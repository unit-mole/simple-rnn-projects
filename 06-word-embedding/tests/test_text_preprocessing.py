from src.text_preprocessing import build_vocabulary, clean_text, encode_tokens, tokenize


def test_clean_and_tokenize_text():
    text = "<b>Quality</b> inspection at test@example.com found defect #42."
    assert clean_text(text) == "quality inspection at found defect"
    assert tokenize(text) == ["quality", "inspection", "at", "found", "defect"]


def test_vocabulary_and_oov_encoding():
    sentences = [["quality", "defect"], ["quality", "inspection"]]
    word_to_index, index_to_word, _ = build_vocabulary(
        sentences, min_frequency=1, max_vocabulary_size=10
    )
    assert word_to_index["<PAD>"] == 0
    assert word_to_index["<UNK>"] == 1
    encoded = encode_tokens(["quality", "unknownword"], word_to_index)
    assert encoded[0] == word_to_index["quality"]
    assert encoded[1] == word_to_index["<UNK>"]
    assert index_to_word[word_to_index["defect"]] == "defect"
