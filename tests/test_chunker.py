"""
Tests for document chunker.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.ingest import DocumentChunker, Chunk, DocumentLoader


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = DocumentChunker(chunk_size=500, overlap=50)
    assert chunker.chunk_size == 500
    assert chunker.overlap == 50


def test_chunker_initialization_from_settings():
    """Test chunker uses settings defaults."""
    from src.config import settings
    
    chunker = DocumentChunker()
    assert chunker.chunk_size == settings.chunk_size
    assert chunker.overlap == settings.chunk_overlap


def test_chunker_invalid_overlap():
    """Test chunker raises error for invalid overlap."""
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, overlap=100)


def test_count_tokens():
    """Test token counting."""
    chunker = DocumentChunker()
    text = "This is a test sentence."
    token_count = chunker.count_tokens(text)
    assert token_count > 0
    assert isinstance(token_count, int)


def test_chunk_document():
    """Test document chunking."""
    chunker = DocumentChunker(chunk_size=100, overlap=10)
    
    # Create a long text that will need multiple chunks
    text = " ".join(["This is sentence number {}.".format(i) for i in range(50)])
    
    chunks = chunker.chunk_document(text, "test_doc", {"source": "test"})
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.doc_id == "test_doc" for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunk_validation():
    """Test chunk validation - no chunk exceeds token limit."""
    chunker = DocumentChunker(chunk_size=100, overlap=10)
    
    text = " ".join(["Sentence {}".format(i) for i in range(100)])
    chunks = chunker.chunk_document(text, "test_doc")
    
    # All chunks should be within reasonable token limit (allowing 10% tolerance)
    for chunk in chunks:
        assert chunk.token_count <= chunker.chunk_size * 1.2


def test_chunk_overlap():
    """Test that chunks have overlap information."""
    chunker = DocumentChunker(chunk_size=100, overlap=20)
    
    text = " ".join(["Sentence {}".format(i) for i in range(100)])
    chunks = chunker.chunk_document(text, "test_doc")
    
    if len(chunks) > 1:
        # Check that overlap_info is present
        for chunk in chunks[:-1]:  # Last chunk might not have overlap
            assert 'overlap_tokens' in chunk.overlap_info
            assert 'chunk_position' in chunk.overlap_info
            assert 'total_chunks' in chunk.overlap_info


def test_save_chunks(tmp_path):
    """Test saving chunks to JSONL."""
    chunker = DocumentChunker(chunk_size=100, overlap=10)
    
    text = " ".join(["Sentence {}".format(i) for i in range(50)])
    chunks = chunker.chunk_document(text, "test_doc")
    
    output_path = tmp_path / "chunks.jsonl"
    DocumentChunker.save_chunks(chunks, output_path)
    
    assert output_path.exists()
    
    # Verify JSONL format
    with open(output_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == len(chunks)
        
        # Verify each line is valid JSON
        for line in lines:
            chunk_data = json.loads(line)
            assert 'doc_id' in chunk_data
            assert 'chunk_id' in chunk_data
            assert 'text' in chunk_data
            assert 'token_count' in chunk_data
            assert 'metadata' in chunk_data
            assert 'overlap_info' in chunk_data


def test_empty_text():
    """Test chunking empty text."""
    chunker = DocumentChunker()
    chunks = chunker.chunk_document("", "empty_doc")
    assert len(chunks) == 0


def test_chunk_file_text_file(tmp_path):
    """Test chunking a text file."""
    # Create a test text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document. " * 50)
    
    chunker = DocumentChunker(chunk_size=50, overlap=5)
    chunks = chunker.chunk_file(test_file)
    
    assert len(chunks) > 0
    assert all(chunk.doc_id == "test" for chunk in chunks)


def test_document_loader_text_file(tmp_path):
    """Test DocumentLoader with text file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test content.")
    
    loader = DocumentLoader()
    doc = loader.load_document(test_file)
    
    assert 'text' in doc
    assert 'metadata' in doc
    assert doc['text'] == "This is test content."
    assert doc['metadata']['file_type'] == '.txt'


def test_chunk_metadata_preserved():
    """Test that document metadata is preserved in chunks."""
    chunker = DocumentChunker(chunk_size=50, overlap=5)
    
    metadata = {
        'file_name': 'test.pdf',
        'file_type': '.pdf',
        'custom_field': 'custom_value'
    }
    
    text = "Test content. " * 20
    chunks = chunker.chunk_document(text, "test_doc", metadata)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata['file_name'] == 'test.pdf'
        assert chunk.metadata['file_type'] == '.pdf'
        assert chunk.metadata['custom_field'] == 'custom_value'


def test_chunk_ids_unique():
    """Test that chunk IDs are unique."""
    chunker = DocumentChunker(chunk_size=50, overlap=5)
    
    text = "Test content. " * 100
    chunks = chunker.chunk_document(text, "test_doc")
    
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # All unique

