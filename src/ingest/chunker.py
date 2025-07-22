import re
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from config.settings import settings


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    SIMPLE = "simple"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    CODE = "code"


@dataclass
class Chunk:
    """A text chunk with metadata."""
    index: int
    content: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]
    overlap_with_previous: bool = False
    overlap_with_next: bool = False


class TextChunker:
    """Intelligent text chunking with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    ):
        self.chunk_size = chunk_size or settings.ingest.chunk_size
        self.chunk_overlap = chunk_overlap or settings.ingest.chunk_overlap
        self.strategy = strategy
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        # Code-specific patterns
        self.code_block_patterns = {
            'function': re.compile(r'^\s*(def|function|class|interface)\s+\w+', re.MULTILINE),
            'block_comment': re.compile(r'/\*.*?\*/', re.DOTALL),
            'line_comment': re.compile(r'//.*?$|#.*?$', re.MULTILINE),
        }
        
        # Markdown patterns
        self.markdown_patterns = {
            'header': re.compile(r'^#+\s+.*$', re.MULTILINE),
            'code_block': re.compile(r'```.*?```', re.DOTALL),
            'list_item': re.compile(r'^\s*[-*+]\s+.*$', re.MULTILINE),
        }
    
    def chunk_text(
        self, 
        text: str, 
        content_type: str = "text/plain",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk text using the configured strategy."""
        if not text or not text.strip():
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Choose chunking strategy based on content type
        strategy = self._select_strategy(content_type)
        
        logger.info(f"Chunking text ({len(text)} chars) using {strategy.value} strategy")
        
        # Apply chunking strategy
        if strategy == ChunkingStrategy.SIMPLE:
            chunks = self._chunk_simple(text)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentences(text)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraphs(text)
        elif strategy == ChunkingStrategy.MARKDOWN:
            chunks = self._chunk_markdown(text)
        elif strategy == ChunkingStrategy.CODE:
            chunks = self._chunk_code(text)
        else:
            chunks = self._chunk_by_sentences(text)  # Default fallback
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata.update(metadata or {})
            chunk.metadata['chunking_strategy'] = strategy.value
            chunk.metadata['chunk_size_config'] = self.chunk_size
            chunk.metadata['overlap_config'] = self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)  # Windows line endings
        text = re.sub(r'\r', '\n', text)    # Mac line endings
        text = re.sub(r'\t', ' ', text)     # Tabs to spaces
        text = re.sub(r' +', ' ', text)     # Multiple spaces to single
        
        # Remove excessive blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _select_strategy(self, content_type: str) -> ChunkingStrategy:
        """Select appropriate chunking strategy based on content type."""
        if 'markdown' in content_type.lower():
            return ChunkingStrategy.MARKDOWN
        elif any(lang in content_type.lower() for lang in ['python', 'javascript', 'java', 'code']):
            return ChunkingStrategy.CODE
        elif 'pdf' in content_type.lower():
            return ChunkingStrategy.PARAGRAPH
        else:
            return self.strategy
    
    def _chunk_simple(self, text: str) -> List[Chunk]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Look back for space within overlap distance
                space_pos = text.rfind(' ', end - self.chunk_overlap, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    index=chunk_index,
                    content=chunk_text,
                    start_char=start,
                    end_char=end,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={'method': 'simple'},
                    overlap_with_previous=start > 0 and chunk_index > 0,
                    overlap_with_next=end < len(text)
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[Chunk]:
        """Chunk by sentences with size limits."""
        sentences = self._split_sentences(text)
        chunks = []
        chunk_index = 0
        
        current_chunk = []
        current_size = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    end_char = start_char + len(chunk_text)
                    chunks.append(Chunk(
                        index=chunk_index,
                        content=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=self._estimate_tokens(chunk_text),
                        metadata={'method': 'sentence', 'sentence_count': len(current_chunk)},
                        overlap_with_previous=chunk_index > 0,
                        overlap_with_next=True
                    ))
                    chunk_index += 1
                
                # Handle overlap - keep last few sentences
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
                start_char = end_char - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(Chunk(
                    index=chunk_index,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={'method': 'sentence', 'sentence_count': len(current_chunk)},
                    overlap_with_previous=chunk_index > 0,
                    overlap_with_next=False
                ))
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[Chunk]:
        """Chunk by paragraphs with size limits."""
        paragraphs = self.paragraph_breaks.split(text)
        chunks = []
        chunk_index = 0
        
        current_chunk = []
        current_size = 0
        start_char = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_size = len(paragraph)
            
            # If paragraph alone exceeds chunk size, split it by sentences
            if paragraph_size > self.chunk_size:
                if current_chunk:
                    # Finalize current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        index=chunk_index,
                        content=chunk_text,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        token_count=self._estimate_tokens(chunk_text),
                        metadata={'method': 'paragraph', 'paragraph_count': len(current_chunk)},
                        overlap_with_previous=chunk_index > 0,
                        overlap_with_next=True
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                large_para_chunks = self._chunk_by_sentences(paragraph)
                for large_chunk in large_para_chunks:
                    large_chunk.index = chunk_index
                    large_chunk.metadata['method'] = 'paragraph_overflow'
                    chunks.append(large_chunk)
                    chunk_index += 1
                
                start_char += paragraph_size + 2  # Account for paragraph breaks
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    index=chunk_index,
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={'method': 'paragraph', 'paragraph_count': len(current_chunk)},
                    overlap_with_previous=chunk_index > 0,
                    overlap_with_next=True
                ))
                chunk_index += 1
                
                # Reset for next chunk
                current_chunk = [paragraph]
                current_size = paragraph_size
                start_char += len(chunk_text) + 2
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                index=chunk_index,
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                token_count=self._estimate_tokens(chunk_text),
                metadata={'method': 'paragraph', 'paragraph_count': len(current_chunk)},
                overlap_with_previous=chunk_index > 0,
                overlap_with_next=False
            ))
        
        return chunks
    
    def _chunk_markdown(self, text: str) -> List[Chunk]:
        """Chunk markdown text respecting structure."""
        # Find major sections (headers)
        sections = self._split_markdown_sections(text)
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_size = len(section)
            
            if section_size <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    index=chunk_index,
                    content=section.strip(),
                    start_char=0,  # Will be adjusted later
                    end_char=section_size,
                    token_count=self._estimate_tokens(section),
                    metadata={'method': 'markdown_section'},
                    overlap_with_previous=chunk_index > 0,
                    overlap_with_next=False
                ))
                chunk_index += 1
            else:
                # Split large section by paragraphs
                section_chunks = self._chunk_by_paragraphs(section)
                for chunk in section_chunks:
                    chunk.index = chunk_index
                    chunk.metadata['method'] = 'markdown_large_section'
                    chunks.append(chunk)
                    chunk_index += 1
        
        return chunks
    
    def _chunk_code(self, text: str) -> List[Chunk]:
        """Chunk code text respecting structure."""
        # Try to split by functions/classes first
        functions = self._find_code_functions(text)
        
        if functions:
            chunks = []
            chunk_index = 0
            
            for func_start, func_end, func_name in functions:
                func_text = text[func_start:func_end]
                
                if len(func_text) <= self.chunk_size:
                    chunks.append(Chunk(
                        index=chunk_index,
                        content=func_text.strip(),
                        start_char=func_start,
                        end_char=func_end,
                        token_count=self._estimate_tokens(func_text),
                        metadata={'method': 'code_function', 'function_name': func_name},
                        overlap_with_previous=False,
                        overlap_with_next=False
                    ))
                    chunk_index += 1
                else:
                    # Split large function by lines
                    func_chunks = self._chunk_by_lines(func_text, func_start)
                    for chunk in func_chunks:
                        chunk.index = chunk_index
                        chunk.metadata['method'] = 'code_large_function'
                        chunk.metadata['function_name'] = func_name
                        chunks.append(chunk)
                        chunk_index += 1
            
            return chunks
        else:
            # Fall back to line-based chunking
            return self._chunk_by_lines(text)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_markdown_sections(self, text: str) -> List[str]:
        """Split markdown by headers."""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            if line.startswith('#') and current_section:
                # New section starts
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _find_code_functions(self, text: str) -> List[tuple]:
        """Find function/class boundaries in code."""
        functions = []
        lines = text.split('\n')
        
        current_func = None
        current_start = 0
        current_indent = 0
        
        for i, line in enumerate(lines):
            # Check for function/class definition
            func_match = self.code_block_patterns['function'].match(line)
            if func_match:
                # Save previous function
                if current_func:
                    functions.append((current_start, i, current_func))
                
                # Start new function
                current_func = func_match.group(0).strip()
                current_start = i
                current_indent = len(line) - len(line.lstrip())
            elif current_func and line.strip():
                # Check if we're still in the same function (by indentation)
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= current_indent and not line.startswith(' '):
                    # Function ended
                    functions.append((current_start, i, current_func))
                    current_func = None
        
        # Add final function
        if current_func:
            functions.append((current_start, len(lines), current_func))
        
        # Convert line numbers to character positions
        char_functions = []
        for start_line, end_line, name in functions:
            start_char = sum(len(lines[i]) + 1 for i in range(start_line))
            end_char = sum(len(lines[i]) + 1 for i in range(end_line))
            char_functions.append((start_char, end_char, name))
        
        return char_functions
    
    def _chunk_by_lines(self, text: str, start_offset: int = 0) -> List[Chunk]:
        """Chunk code by lines with size limits."""
        lines = text.split('\n')
        chunks = []
        chunk_index = 0
        
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # Include newline
            
            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append(Chunk(
                    index=chunk_index,
                    content=chunk_text,
                    start_char=start_offset,
                    end_char=start_offset + len(chunk_text),
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={'method': 'code_lines', 'line_count': len(current_chunk)},
                    overlap_with_previous=chunk_index > 0,
                    overlap_with_next=True
                ))
                chunk_index += 1
                
                # Add overlap
                overlap_lines = current_chunk[-min(3, len(current_chunk)):]
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                start_offset += len('\n'.join(current_chunk[:-len(overlap_lines)]))
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(Chunk(
                index=chunk_index,
                content=chunk_text,
                start_char=start_offset,
                end_char=start_offset + len(chunk_text),
                token_count=self._estimate_tokens(chunk_text),
                metadata={'method': 'code_lines', 'line_count': len(current_chunk)},
                overlap_with_previous=chunk_index > 0,
                overlap_with_next=False
            ))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on configured overlap size."""
        overlap_chars = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_chars + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_chars += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)."""
        return max(1, len(text) // 4)


# Global chunker instance
text_chunker = TextChunker()