#!/usr/bin/env python3
"""
Generate a manifest CSV for OpenAlex papers in PaperQA format.

This script scans the OpenAlex directory structure and creates a manifest CSV
with document metadata extracted from the OpenAlex JSON files.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def clean_doi(doi: str) -> str:
    """Clean DOI by removing URL prefixes."""
    if doi.startswith("https://doi.org/"):
        return doi[16:]
    elif doi.startswith("http://dx.doi.org/"):
        return doi[18:]
    return doi


def extract_authors(authorships: List[Dict[str, Any]]) -> List[str]:
    """Extract author names from OpenAlex authorships."""
    authors = []
    for authorship in authorships:
        author = authorship.get("author", {})
        display_name = author.get("display_name", "")
        if display_name:
            authors.append(display_name)
    return authors


def format_citation(metadata: Dict[str, Any]) -> str:
    """Format a basic citation from OpenAlex metadata."""
    authors = extract_authors(metadata.get("authorships", []))
    title = metadata.get("title", "")
    year = metadata.get("publication_year", "")
    
    # Get journal name from primary location
    journal = ""
    primary_location = metadata.get("primary_location", {})
    if primary_location:
        source = primary_location.get("source", {})
        if source:
            journal = source.get("display_name", "")
    
    # Format citation
    if len(authors) == 0:
        author_str = "Unknown authors"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    else:
        author_str = f"{authors[0]} et al."
    
    citation_parts = []
    if author_str:
        citation_parts.append(author_str)
    if title:
        citation_parts.append(title)
    if journal and year:
        citation_parts.append(f"{journal}, {year}")
    elif journal:
        citation_parts.append(journal)
    elif year:
        citation_parts.append(str(year))
    
    return ". ".join(citation_parts) + "." if citation_parts else "Unknown citation"


def create_docname(metadata: Dict[str, Any]) -> str:
    """Create a docname from metadata."""
    authors = extract_authors(metadata.get("authorships", []))
    year = metadata.get("publication_year", "")
    title = metadata.get("title", "")
    
    # Get first author's last name
    first_author_last = ""
    if authors:
        first_author = authors[0]
        # Try to extract last name (simple heuristic)
        name_parts = first_author.split()
        if name_parts:
            first_author_last = name_parts[-1]
    
    # Create a clean docname
    if first_author_last and year:
        docname = f"{first_author_last}{year}"
    elif first_author_last:
        docname = first_author_last
    elif title:
        # Use first few words of title if no author
        clean_title = re.sub(r'[^\w\s]', '', title)
        words = clean_title.split()[:3]
        docname = "".join(words)
    else:
        docname = "Unknown"
    
    # Clean docname
    docname = re.sub(r'[^\w]', '', docname)
    return docname


def get_pdf_url(metadata: Dict[str, Any]) -> Optional[str]:
    """Extract PDF URL from OpenAlex metadata."""
    # Check best OA location first
    best_oa = metadata.get("best_oa_location", {})
    if best_oa and best_oa.get("pdf_url"):
        return best_oa["pdf_url"]
    
    # Check other locations
    for location in metadata.get("locations", []):
        if location.get("pdf_url"):
            return location["pdf_url"]
    
    return None


def get_issn(metadata: Dict[str, Any]) -> Optional[str]:
    """Extract ISSN from OpenAlex metadata."""
    primary_location = metadata.get("primary_location", {})
    if primary_location:
        source = primary_location.get("source", {})
        if source:
            issn_list = source.get("issn")
            if issn_list and isinstance(issn_list, list) and len(issn_list) > 0:
                return issn_list[0]
            elif isinstance(issn_list, str):
                return issn_list
            
            # Try issn_l as fallback
            issn_l = source.get("issn_l")
            if issn_l:
                return issn_l
    
    return None


def process_metadata_file(metadata_path: Path, base_dir: Path) -> Optional[Dict[str, Any]]:
    """Process a single metadata JSON file and return manifest entry."""    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Get the directory name for file location
    subdir = metadata_path.parent
    subdir_name = subdir.name
    
    # Look for PDF file in the same directory
    pdf_file = None
    for file in subdir.glob("*.pdf"):
        pdf_file = file.name
        break
    
    if not pdf_file:
        pdf_file = f"{subdir_name}.pdf"  # Assume standard naming
    
    # Create relative file location
    file_location = f"{subdir_name}/{pdf_file}"
    
    # Extract metadata fields
    authors = extract_authors(metadata.get("authorships", []))
    
    # Parse publication date
    pub_date = metadata.get("publication_date")
    publication_date = None
    if pub_date:
        publication_date = datetime.fromisoformat(pub_date).isoformat()
    
    # Get biblio info
    biblio = metadata.get("biblio", {}) or {}
    
    # Create manifest entry
    entry = {
        "file_location": file_location,
        "title": metadata.get("title", ""),
        "doi": clean_doi(metadata.get("doi", "")) if metadata.get("doi") else "",
        "authors": authors,
        "year": metadata.get("publication_year", ""),
        "publication_date": publication_date or "",
        "journal": (metadata.get("primary_location", {}).get("source") or {}).get("display_name", ""),
        "publisher": (metadata.get("primary_location", {}).get("source") or {}).get("host_organization_name", ""),
        "volume": biblio.get("volume", ""),
        "issue": biblio.get("issue", ""),
        "pages": f"{biblio.get('first_page', '')}-{biblio.get('last_page', '')}" if biblio.get('first_page') and biblio.get('last_page') else biblio.get('first_page', ""),
        "issn": get_issn(metadata) or "",
        "url": metadata.get("id", ""),  # OpenAlex URL
        "pdf_url": get_pdf_url(metadata) or "",
        "citation_count": metadata.get("cited_by_count", ""),
        "is_retracted": metadata.get("is_retracted", False),
        "doc_id": metadata.get("id", "").split("/")[-1] if metadata.get("id") else "",  # Extract ID from URL
        "citation": format_citation(metadata),
        "docname": create_docname(metadata),
        "key": create_docname(metadata),
        "bibtex_type": "article",  # Most OpenAlex entries are articles
    }
    
    return entry


def generate_manifest(openalex_dir: str, output_file: str = "manifest.csv"):
    """Generate manifest CSV from OpenAlex directory structure."""
    base_dir = Path(openalex_dir)
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory {openalex_dir} does not exist")
    
    manifest_entries = []
    processed_count = 0
    
    print(f"Scanning {base_dir} for metadata files...")
    
    # Find all metadata JSON files
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("meta_"):
            metadata_file = subdir / f"{subdir.name}_metadata.json"
            
            if metadata_file.exists():
                entry = process_metadata_file(metadata_file, base_dir)
                manifest_entries.append(entry)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
    
    print(f"Processed {processed_count} files")
    print(f"Creating manifest with {len(manifest_entries)} entries...")
    
    # Define field order for CSV (file_location and doi first as per DocDetails)
    fieldnames = [
        "file_location", "doi", "title", "authors", "year", "publication_date", 
        "journal", "publisher", "volume", "issue", "pages", "issn", "url", 
        "pdf_url", "citation_count", "is_retracted", "doc_id", "citation", 
        "docname", "key", "bibtex_type"
    ]
    
    # Write CSV
    output_path = Path(output_file)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_entries)
    
    print(f"Manifest saved to {output_path}")
    print(f"Total entries: {len(manifest_entries)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate manifest CSV from OpenAlex metadata")
    parser.add_argument("openalex_dir", help="Path to OpenAlex directory containing subdirectories with metadata")
    parser.add_argument("-o", "--output", default="manifest.csv", help="Output CSV file (default: manifest.csv)")
    
    args = parser.parse_args()
    
    try:
        generate_manifest(args.openalex_dir, args.output)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
