"""Multiprocessing utilities for PDF parsing to work around thread-safety issues."""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor
from functools import partial
from typing import Any

from paperqa.types import ParsedText


def _parse_pdf_worker(
    parser_func: Callable[..., ParsedText],
    path: str | os.PathLike,
    **kwargs: Any,
) -> ParsedText:
    """Worker function to parse a single PDF in a separate process.
    
    Args:
        parser_func: The PDF parsing function to use
        path: Path to the PDF file
        **kwargs: Additional arguments to pass to the parser
        
    Returns:
        ParsedText: The parsed PDF content
    """
    try:
        return parser_func(path, **kwargs)
    except Exception as e:
        # Re-raise with more context to help with debugging
        raise type(e)(f"PDF parsing failed for {path}: {e}") from e


class PDFProcessingPool:
    """A process pool for parsing PDFs in parallel while avoiding thread-safety issues."""
    
    def __init__(self, max_workers: int | None = None):
        """Initialize the PDF processing pool.
        
        Args:
            max_workers: Maximum number of worker processes. If None, defaults to 
                        min(4, number of CPU cores)
        """
        if max_workers is None:
            # Conservative default: use at most 4 processes, but don't exceed CPU count
            max_workers = os.cpu_count()
        
        self.max_workers = max_workers
        self._pool: ProcessPoolExecutor | None = None
    
    def __enter__(self) -> PDFProcessingPool:
        """Enter the context manager and create the process pool."""
        self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and shutdown the process pool."""
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
    
    def parse_pdf(
        self,
        parser_func: Callable[..., ParsedText],
        path: str | os.PathLike,
        **kwargs: Any,
    ) -> ParsedText:
        """Parse a single PDF using the process pool.
        
        Args:
            parser_func: The PDF parsing function to use
            path: Path to the PDF file
            **kwargs: Additional arguments to pass to the parser
            
        Returns:
            ParsedText: The parsed PDF content
        """
        if self._pool is None:
            raise RuntimeError("PDFProcessingPool must be used as a context manager")
        
        # Submit the parsing task to the process pool
        future = self._pool.submit(_parse_pdf_worker, parser_func, path, **kwargs)
        return future.result()
    
    def parse_pdfs_batch(
        self,
        parser_func: Callable[..., ParsedText],
        paths: list[str | os.PathLike],
        kwargs_list: list[dict[str, Any]] | None = None,
    ) -> list[ParsedText]:
        """Parse multiple PDFs in parallel using the process pool.
        
        Args:
            parser_func: The PDF parsing function to use
            paths: List of paths to PDF files
            kwargs_list: List of keyword arguments for each PDF. If None, 
                        empty dict is used for all.
            
        Returns:
            List of ParsedText objects in the same order as input paths
        """
        if self._pool is None:
            raise RuntimeError("PDFProcessingPool must be used as a context manager")
        
        if kwargs_list is None:
            kwargs_list = [{}] * len(paths)
        
        if len(paths) != len(kwargs_list):
            raise ValueError("paths and kwargs_list must have the same length")
        
        # Submit all parsing tasks
        future_to_index = {}
        for i, (path, kwargs) in enumerate(zip(paths, kwargs_list, strict=True)):
            future = self._pool.submit(_parse_pdf_worker, parser_func, path, **kwargs)
            future_to_index[future] = i
        
        # Collect results in order
        results = [None] * len(paths)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
        
        return results


# Global pool instance for reuse across the application
_global_pdf_pool: PDFProcessingPool | None = None


def get_pdf_processing_pool(max_workers: int | None = None) -> PDFProcessingPool:
    """Get or create a global PDF processing pool.
    
    Args:
        max_workers: Maximum number of worker processes. Only used when creating
                    a new pool.
    
    Returns:
        PDFProcessingPool: The global pool instance
    """
    global _global_pdf_pool
    if _global_pdf_pool is None:
        _global_pdf_pool = PDFProcessingPool(max_workers=max_workers)
    return _global_pdf_pool


def parse_pdf_with_multiprocessing(
    parser_func: Callable[..., ParsedText],
    path: str | os.PathLike,
    use_multiprocessing: bool = True,
    max_workers: int | None = None,
    **kwargs: Any,
) -> ParsedText:
    """Parse a PDF with optional multiprocessing support.
    
    Args:
        parser_func: The PDF parsing function to use
        path: Path to the PDF file
        use_multiprocessing: Whether to use multiprocessing. If False, calls
                           parser_func directly.
        max_workers: Maximum number of worker processes for the pool
        **kwargs: Additional arguments to pass to the parser
        
    Returns:
        ParsedText: The parsed PDF content
        
    Raises:
        Any exception from the multiprocessing pool or PDF parser.
        If multiprocessing fails, the exception propagates so users can
        disable multiprocessing or fix the underlying issue.
    """
    if not use_multiprocessing:
        return parser_func(path, **kwargs)
    
    # Use the global shared pool instead of creating a new one for each PDF
    global _global_pdf_pool
    if _global_pdf_pool is None:
        _global_pdf_pool = PDFProcessingPool(max_workers=max_workers)
        _global_pdf_pool.__enter__()  # Initialize the pool
    
    return _global_pdf_pool.parse_pdf(parser_func, path, **kwargs)
