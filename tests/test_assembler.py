from audio_transcript.assembler import combine


def test_combine_three_chunks_separated_by_double_newline():
    chunks = ["Senhor, abençoe.", "Ore por minha mãe.", "Obrigado a todos."]
    result = combine(chunks)
    assert result == "Senhor, abençoe.\n\nOre por minha mãe.\n\nObrigado a todos."


def test_combine_trims_whitespace_in_each_chunk():
    chunks = ["  hello  ", "\nworld\n"]
    result = combine(chunks)
    assert result == "hello\n\nworld"


def test_combine_drops_empty_chunks():
    chunks = ["one", "", "  ", "two"]
    result = combine(chunks)
    assert result == "one\n\ntwo"


def test_combine_single_chunk_no_trailing_separator():
    result = combine(["only one"])
    assert result == "only one"


def test_combine_empty_input_returns_empty_string():
    assert combine([]) == ""
