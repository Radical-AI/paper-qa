#!/usr/bin/env python3
"""
Script to generate a manifest.csv for the indexer from ChemRxiv metadata files.

This script crawls a directory structure containing paper subdirectories,
each with a metadata JSON file and PDF, extracts relevant metadata,
and creates a manifest CSV file for the paperqa indexer.
"""

import asyncio
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles


@dataclass
class PaperMetadata:
    """Container for extracted paper metadata matching DocDetails schema."""
    file_location: str
    doi: str
    title: str
    authors: List[str]
    publication_date: Optional[str] = None  # Will be converted to datetime format
    pdf_url: Optional[str] = None
    doc_id: Optional[str] = None
    other: Optional[Dict[str, Any]] = None  # For non-standard fields like abstract, categories, keywords


def extract_metadata_from_json(json_path: Path, pdf_path: Path, base_dir: Path) -> Optional[PaperMetadata]:
    """
    Extract relevant metadata from a single JSON metadata file.
    
    Args:
        json_path: Path to the metadata JSON file
        pdf_path: Path to the corresponding PDF file
        base_dir: Base directory for calculating relative paths
        
    Returns:
        PaperMetadata object if successful, None if failed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Calculate relative file location from base directory
        rel_pdf_path = pdf_path.relative_to(base_dir)
        
        # Extract required fields
        file_location = str(rel_pdf_path)
        doi = data.get('doi', '')
        title = data.get('title', '')
        
        # Extract optional fields that map to DocDetails
        authors = data.get('authors', [])
        if isinstance(authors, list):
            authors = [str(author) for author in authors]
        else:
            authors = []
            
        publication_date = data.get('published_date')  # Keep as string, will be parsed by paperqa
        pdf_url = data.get('pdf_url')
        doc_id = data.get('id')  # ChemRxiv 'id' maps to DocDetails 'doc_id'
        
        # Put non-standard fields into 'other' dict
        other_fields = {}
        for field in ['abstract', 'categories', 'keywords', 'submitted_date', 'version', 'status', 'abstract_views', 'downloads']:
            if field in data:
                other_fields[field] = data[field]
        
        # Validate required fields
        if not file_location or not doi or not title:
            print(f"Warning: Missing required fields in {json_path}")
            return None
            
        return PaperMetadata(
            file_location=file_location,
            doi=doi,
            title=title,
            authors=authors,
            publication_date=publication_date,
            pdf_url=pdf_url,
            doc_id=doc_id,
            other=other_fields if other_fields else None
        )
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None


async def process_single_directory(subdir_path: Path, base_dir: Path) -> Optional[PaperMetadata]:
    """
    Process a single subdirectory to extract metadata.
    
    Args:
        subdir_path: Path to the subdirectory
        base_dir: Base directory for calculating relative paths
        
    Returns:
        PaperMetadata object if successful, None if failed
    """
    try:
        # Find metadata JSON and PDF files
        json_files = list(subdir_path.glob("*_metadata.json"))
        pdf_files = list(subdir_path.glob("*.pdf"))
        
        if not json_files:
            print(f"Warning: No metadata JSON found in {subdir_path}")
            return None
            
        if not pdf_files:
            print(f"Warning: No PDF found in {subdir_path}")
            return None
            
        json_path = json_files[0]  # Use first metadata file found
        pdf_path = pdf_files[0]    # Use first PDF file found
        
        # Extract metadata using sync function in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            metadata = await loop.run_in_executor(
                executor, extract_metadata_from_json, json_path, pdf_path, base_dir
            )
            
        return metadata
        
    except Exception as e:
        print(f"Error processing directory {subdir_path}: {e}")
        return None


async def crawl_directory_parallel(base_dir: Path, max_workers: int = 50) -> List[PaperMetadata]:
    """
    Crawl the directory structure in parallel to extract metadata from all papers.
    
    Args:
        base_dir: Base directory containing paper subdirectories, or a single paper directory
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of PaperMetadata objects
    """
    print(f"Starting to crawl directory: {base_dir}")
    
    # Check if this is a single paper directory (contains metadata JSON and PDF)
    json_files = list(base_dir.glob("*_metadata.json"))
    pdf_files = list(base_dir.glob("*.pdf"))
    
    if json_files and pdf_files:
        # This is a single paper directory
        print("Detected single paper directory")
        parent_dir = base_dir.parent
        metadata = await process_single_directory(base_dir, parent_dir)
        return [metadata] if metadata else []
    
    # This is a parent directory containing multiple paper subdirectories
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    print(f"Found {len(subdirs)} subdirectories to process")
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_semaphore(subdir: Path) -> Optional[PaperMetadata]:
        async with semaphore:
            return await process_single_directory(subdir, base_dir)
    
    # Process all subdirectories in parallel
    print(f"Processing with {max_workers} concurrent workers...")
    tasks = [process_with_semaphore(subdir) for subdir in subdirs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    metadata_list = []
    error_count = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception in processing {subdirs[i]}: {result}")
            error_count += 1
        elif result is not None:
            metadata_list.append(result)
        else:
            error_count += 1
    
    print(f"Successfully processed {len(metadata_list)} papers")
    print(f"Encountered {error_count} errors")
    
    return metadata_list


async def write_manifest_csv(metadata_list: List[PaperMetadata], output_path: Path) -> None:
    """
    Write the manifest CSV file.
    
    Args:
        metadata_list: List of PaperMetadata objects
        output_path: Path to write the CSV file
    """
    print(f"Writing manifest to: {output_path}")
    
    # Define CSV columns based on actual DocDetails fields
    # Following DocDetails.CSV_FIELDS_UP_FRONT pattern (line 691): doi, file_location first
    fieldnames = [
        'doi',
        'file_location',
        'title',
        'authors',
        'publication_date',
        'pdf_url',
        'doc_id',
        'other'
    ]
    
    async with aiofiles.open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Write CSV header
        header_line = ','.join(fieldnames) + '\n'
        await csvfile.write(header_line)
        
        # Write data rows
        for metadata in metadata_list:
            # Convert authors list to JSON string for proper parsing
            authors_json = json.dumps(metadata.authors) if metadata.authors else '[]'
            
            # Convert other dict to JSON string
            other_json = json.dumps(metadata.other) if metadata.other else '{}'
            
            row_data = [
                metadata.doi,
                metadata.file_location,
                metadata.title,
                authors_json,
                metadata.publication_date or '',
                metadata.pdf_url or '',
                metadata.doc_id or '',
                other_json
            ]
            
            # Escape and quote fields that contain commas or quotes
            escaped_row = []
            for field in row_data:
                field_str = str(field)
                if ',' in field_str or '"' in field_str or '\n' in field_str:
                    # Escape quotes by doubling them and wrap in quotes
                    escaped_field = '"' + field_str.replace('"', '""') + '"'
                    escaped_row.append(escaped_field)
                else:
                    escaped_row.append(field_str)
            
            row_line = ','.join(escaped_row) + '\n'
            await csvfile.write(row_line)
    
    print(f"Manifest CSV written with {len(metadata_list)} entries")


async def main():
    """Main function to orchestrate the manifest creation process."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python create_manifest.py <directory_path> [output_path] [max_workers]")
        print("Example: python create_manifest.py /data/kai/radical-scratch/chemrxiv")
        sys.exit(1)
    
    directory_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else directory_path / "manifest.csv"
    max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    # Validate input directory
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist")
        sys.exit(1)
    
    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a directory")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Input directory: {directory_path}")
    print(f"  Output file: {output_path}")
    print(f"  Max workers: {max_workers}")
    print()
    
    try:
        # Crawl directory and extract metadata
        metadata_list = await crawl_directory_parallel(directory_path, max_workers)
        
        if not metadata_list:
            print("No valid metadata found. Exiting.")
            sys.exit(1)
        
        # Write manifest CSV
        await write_manifest_csv(metadata_list, output_path)
        
        print(f"\nManifest creation completed successfully!")
        print(f"Output: {output_path}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
