"""
Custom text splitter to replace langchain RecursiveCharacterTextSplitter.
"""

from typing import Callable, List, Optional


class RecursiveCharacterTextSplitter:
    """Splits text recursively by characters."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Optional[Callable[[str], int]] = None,
        add_start_index: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = (
            separators if separators else ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.length_function = length_function if length_function else len
        self.add_start_index = add_start_index

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not text:
            return []

        # Start with the full text
        splits = [text]

        # Try each separator in order
        for separator in self.separators:
            if not separator:
                # Last separator - split by character
                new_splits = []
                for split in splits:
                    if self.length_function(split) <= self.chunk_size:
                        new_splits.append(split)
                    else:
                        # Split by character
                        for i in range(
                            0, len(split), self.chunk_size - self.chunk_overlap
                        ):
                            chunk = split[i : i + self.chunk_size]
                            if chunk:
                                new_splits.append(chunk)
                splits = new_splits
                break

            new_splits = []
            for split in splits:
                if self.length_function(split) <= self.chunk_size:
                    new_splits.append(split)
                else:
                    # Split by separator
                    parts = split.split(separator)
                    current_chunk = ""
                    for part in parts:
                        part_with_sep = part if not current_chunk else separator + part
                        if (
                            self.length_function(current_chunk + part_with_sep)
                            <= self.chunk_size
                        ):
                            current_chunk += part_with_sep
                        else:
                            if current_chunk:
                                new_splits.append(current_chunk)
                            current_chunk = part_with_sep
                    if current_chunk:
                        new_splits.append(current_chunk)
            splits = new_splits

            # If all splits are small enough, we're done
            if all(self.length_function(s) <= self.chunk_size for s in splits):
                break

        # Apply overlap
        if self.chunk_overlap > 0 and len(splits) > 1:
            overlapped_splits = []
            for i, split in enumerate(splits):
                if i == 0:
                    overlapped_splits.append(split)
                else:
                    # Add overlap from previous chunk
                    prev_chunk = splits[i - 1]
                    overlap_text = (
                        prev_chunk[-self.chunk_overlap :]
                        if len(prev_chunk) > self.chunk_overlap
                        else prev_chunk
                    )
                    overlapped_splits.append(overlap_text + split)
            splits = overlapped_splits

        return splits

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List:
        """Create Document objects from texts."""
        from tools.document import Document

        all_docs = []
        metadatas = metadatas if metadatas else [{}] * len(texts)

        for text, metadata in zip(texts, metadatas):
            splits = self.split_text(text)
            for i, split in enumerate(splits):
                doc_metadata = metadata.copy()
                if self.add_start_index:
                    # Prefer locating the split without the overlap prefix when possible.
                    start_idx = text.find(split)
                    if start_idx == -1 and split:
                        # Overlap can make the full chunk not appear as a contiguous substring.
                        start_idx = text.find(split[: min(64, len(split))])
                    if start_idx != -1:
                        end_idx = min(len(text), start_idx + len(split))
                        doc_metadata["start_index"] = start_idx
                        # 1-based line numbers in the source text (more intuitive than char offsets).
                        doc_metadata["start_line"] = text.count("\n", 0, start_idx) + 1
                        doc_metadata["end_line"] = text.count("\n", 0, end_idx) + 1
                all_docs.append(Document(page_content=split, metadata=doc_metadata))

        return all_docs
