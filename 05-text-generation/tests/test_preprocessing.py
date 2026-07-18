from src.data_preprocessing import chronological_text_split, normalize_text
from src.sequence_generation import create_input_target_pairs
from src.text_preprocessing import CharacterVocabulary


def test_normalize_text_preserves_useful_punctuation():
    value = normalize_text("Hello,   world!\r\n\r\nNext line.")
    assert "Hello, world!" in value
    assert "Next line." in value


def test_chronological_split_has_no_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    split = chronological_text_split(text, validation_fraction=0.2)
    assert split.train_text + split.validation_text == text


def test_sequence_shapes():
    vocabulary = CharacterVocabulary.fit("abcdefg" * 20)
    encoded = vocabulary.encode("abcdefg" * 20)
    X, y = create_input_target_pairs(encoded, sequence_length=10, step=2)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10
