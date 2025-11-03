

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

from src.config import settings

logger = logging.getLogger(__name__)


# sort of DTO for a chunk
@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    doc_id: str
    chunk_id: str
    text: str
    token_count: int
    metadata: Dict[str, Any]
    overlap_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization."""
        return asdict(self)


class DocumentChunker:
  # we create chunks so we don't have to use entire document
#   LLMs need low cost chunking
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        tokenizer: str = "tiktoken",
        encoding_name: str = "cl100k_base"  # Default for GPT-4 and text-embedding-3
    ):
        # initializes chunker with default values from settings
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.tokenizer = tokenizer
        self.encoding_name = encoding_name

        # there are many ways to tokenize based on what we give this tokenizes it
        
        # overlap should be less than chunk size
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")
        
        # Initialize tokenizer
        self._tokenizer_obj = None
        if self.tokenizer == "tiktoken":
            try:
                import tiktoken
                self._tokenizer_obj = tiktoken.get_encoding(encoding_name)
            except ImportError:
                logger.warning("tiktoken not available, falling back to simple character-based chunking")
                self.tokenizer = "simple"
        elif self.tokenizer == "transformers":
            try:
                from transformers import AutoTokenizer
                # Use a default tokenizer if not specified
                self._tokenizer_obj = AutoTokenizer.from_pretrained("gpt2")
            except ImportError:
                logger.warning("transformers not available, falling back to simple chunking")
                self.tokenizer = "simple"
    
    def count_tokens(self, text: str) -> int:
        # counts tokens based on text given
        if self.tokenizer == "tiktoken" and self._tokenizer_obj:
            return len(self._tokenizer_obj.encode(text))
        elif self.tokenizer == "transformers" and self._tokenizer_obj:
            return len(self._tokenizer_obj.encode(text))
        else:
            # fallback: approximate tokens as ~4 characters per token
            return len(text) // 4
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
# chunks document 
        if not text or not text.strip():
            logger.warning(f"Empty text for doc_id: {doc_id}")
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split text into sentences (simple approach)
        # For better results, could use nltk or spaCy
        sentences = self._split_into_sentences(text)
        
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = 0
        last_chunk_end_sentence_idx = 0
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # sentence should not exceed chunk size
            if current_chunk_tokens + sentence_tokens > self.chunk_size and current_chunk_text:
                # Save current chunk
                overlap_text = self._get_overlap_text(
                    sentences,
                    last_chunk_end_sentence_idx,
                    sentence_idx
                )
                
                chunk = Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    text=current_chunk_text.strip(),
                    token_count=current_chunk_tokens,
                    metadata=metadata.copy(),
                    overlap_info={
                        'overlap_tokens': self.count_tokens(overlap_text),
                        'overlap_text': overlap_text,
                        'chunk_position': chunk_index,
                        'total_chunks': None  # Will update later
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    sentences,
                    last_chunk_end_sentence_idx,
                    sentence_idx
                )
                current_chunk_text = " ".join(overlap_sentences) + " " + sentence
                current_chunk_tokens = self.count_tokens(current_chunk_text)
                last_chunk_end_sentence_idx = max(0, sentence_idx - len(overlap_sentences))
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + sentence
                else:
                    current_chunk_text = sentence
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk if there's remaining text
        if current_chunk_text.strip():
            chunk = Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                text=current_chunk_text.strip(),
                token_count=current_chunk_tokens,
                metadata=metadata.copy(),
                overlap_info={
                    'overlap_tokens': 0,
                    'overlap_text': '',
                    'chunk_position': chunk_index,
                    'total_chunks': None
                }
            )
            chunks.append(chunk)
        
        # Update total_chunks in overlap_info
        for chunk in chunks:
            chunk.overlap_info['total_chunks'] = len(chunks)
        
        # Validate chunks
        self._validate_chunks(chunks)
        
        return chunks
    def _split_into_sentences(self, text: str) -> List[str]:
        # normal approach but i can later try different approaches
        # Simple sentence splitting by periods, exclamation, question marks
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        start_idx: int,
        end_idx: int
    ) -> List[str]:
        # Calculate how many tokens we want in overlap
        # WE WANT OVERLAP FOR BELOW REASON:
        # Benefits:
        # - Context continuity: Each chunk has some context from the previous one
        # - Better retrieval: No information loss at boundaries
        # - Smoother embeddings: Related sentences stay together
        
        target_overlap_tokens = self.overlap
        
        overlap_sentences = []
        overlap_tokens = 0
        
        # Work backwards from end_idx to find sentences that fit in overlap
        # Repeats last few sentences from previous chunk
        for i in range(end_idx - 1, max(-1, start_idx - 1), -1):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            if overlap_tokens + sentence_tokens <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _get_overlap_text(
        self,
        sentences: List[str],
        start_idx: int,
        end_idx: int
    ) -> str:
        # gets overlap text from sentences
        overlap_sentences = self._get_overlap_sentences(sentences, start_idx, end_idx)
        return " ".join(overlap_sentences)
    
    def _validate_chunks(self, chunks: List[Chunk]):
        # validates chunks: checks token limit and overlap continuity
        for i, chunk in enumerate(chunks):
            # token count check
            if chunk.token_count > self.chunk_size * 1.1:  # Allow 10% tolerance
                logger.warning(
                    f"Chunk {chunk.chunk_id} exceeds token limit: "
                    f"{chunk.token_count} > {self.chunk_size}"
                )
            
            # check overlap continuity between chunks
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_text_end = prev_chunk.text[-100:]  # Last 100 chars
                current_text_start = chunk.text[:100]  # First 100 chars
                
                # check if overlap text matches between chunks
                if chunk.overlap_info.get('overlap_tokens', 0) > 0:
                    overlap_text = chunk.overlap_info.get('overlap_text', '')
                    if overlap_text and overlap_text not in prev_chunk.text:
                        logger.warning(
                            f"Overlap discontinuity detected between "
                            f"{prev_chunk.chunk_id} and {chunk.chunk_id}"
                        )
    
    def chunk_file(self, file_path: Path) -> List[Chunk]:
        # loads and chunks a file
        from .document_loader import DocumentLoader
        
        loader = DocumentLoader()
        doc = loader.load_document(file_path)
        
        doc_id = Path(file_path).stem
        return self.chunk_document(
            doc['text'],
            doc_id,
            doc['metadata']
        )
    
    @staticmethod
    def save_chunks(chunks: List[Chunk], output_path: Path):
        # saves chunks to JSONL file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

