from chunker import (
    parse_sections,
    split_long_text,
    create_chunk_id,
)


class TestParseSections:
    def test_single_heading(self):
        md = "# Returns\n\n30-day window."
        sections = parse_sections(md)
        assert len(sections) >= 1
        assert sections[0]["heading"] == "Returns"

    def test_nested_headings(self):
        md = "# Policy\n## Returns\n\n30 days.\n## Shipping\n\n5 days."
        sections = parse_sections(md)
        headings = [s["heading"] for s in sections]
        assert any("Returns" in h for h in headings)
        assert any("Shipping" in h for h in headings)

    def test_body_content_preserved(self):
        md = "# FAQ\n\nHow do I return an item?"
        sections = parse_sections(md)
        bodies = " ".join(s["body"] for s in sections)
        assert "return an item" in bodies

    def test_empty_document(self):
        sections = parse_sections("")
        assert len(sections) <= 1

    def test_no_headings(self):
        md = "Just plain text without any headings."
        sections = parse_sections(md)
        assert len(sections) >= 1


class TestSplitLongText:
    def test_short_text_unchanged(self):
        text = "Short text."
        result = split_long_text(text, chunk_size=100, chunk_overlap=20)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_split(self):
        text = "word " * 200
        result = split_long_text(text, chunk_size=100, chunk_overlap=20)
        assert len(result) > 1

    def test_chunk_size_validation(self):
        try:
            split_long_text("test", chunk_size=10, chunk_overlap=20)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_empty_text(self):
        result = split_long_text("")
        assert result == [] or result == [""]

    def test_chunks_have_content(self):
        text = "abcdefghij " * 100
        result = split_long_text(text, chunk_size=100, chunk_overlap=20)
        for chunk in result:
            assert len(chunk.strip()) > 0


class TestCreateChunkId:
    def test_deterministic(self):
        id1 = create_chunk_id("test.md", 0)
        id2 = create_chunk_id("test.md", 0)
        assert id1 == id2

    def test_different_index_different_id(self):
        id1 = create_chunk_id("test.md", 0)
        id2 = create_chunk_id("test.md", 1)
        assert id1 != id2

    def test_different_file_different_id(self):
        id1 = create_chunk_id("a.md", 0)
        id2 = create_chunk_id("b.md", 0)
        assert id1 != id2

    def test_format(self):
        chunk_id = create_chunk_id("test.md", 5)
        assert chunk_id.startswith("test.md-5-")
        assert len(chunk_id.split("-")) >= 3
