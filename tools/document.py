"""
Custom Document class to replace langchain Document.
"""
from typing import Dict, Any, Optional


class Document:
    """A simple document class with page_content and metadata."""
    
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
    
    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

