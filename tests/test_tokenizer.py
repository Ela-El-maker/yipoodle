from src.tokenizer import CharTokenizer


def test_tokenizer_roundtrip() -> None:
    text = "abc cab"
    tok = CharTokenizer.from_text(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
    assert tok.vocab_size == len(set(text))
