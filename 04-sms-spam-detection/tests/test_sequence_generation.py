import numpy as np
from src.sequence_generation import pad_sequence,texts_to_padded_sequences
from src.text_preprocessing import VocabularyTokenizer

def test_post_padding_and_truncation():
    assert pad_sequence([2,3,4],5).tolist()==[2,3,4,0,0]
    assert pad_sequence([2,3,4,5],2).tolist()==[2,3]

def test_batch_sequence_shape():
    tokenizer=VocabularyTokenizer(20).fit(["free prize","see you soon"])
    array=texts_to_padded_sequences(tokenizer,["free prize","unknown word"],6)
    assert array.shape==(2,6)
    assert array.dtype==np.int32
