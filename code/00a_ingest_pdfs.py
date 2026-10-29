"""
Convert ESG / Sustainability Report PDFs (2025-2026) into the exact Markdown
format used by the corpus in data/raw/{docID}_{companyID}_{YEAR}.md.
"""

import argparse
import json
import os
from pathlib import Path
import re
import pymupdf as fitz  # PyMuPDF


def pdf_to_corpus_markdown(pdf_path: Path, doc_id: int) -> str:
    """Extract page-by-page Markdown matching data/raw/*.md format."""
    doc = fitz.open(pdf_path)
    parts = [f"# Document ID: {doc_id}\n"]

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        # Sort top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        page_lines = []
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            # Clean control characters
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
            # Format short title-like blocks as ## headings to match existing corpus style
            if "\n" not in text and len(text) < 80 and not text.endswith("."):
                page_lines.append(f"## {text}")
            else:
                page_lines.append(text)

        if page_idx > 0:
            parts.append(f"\n---\n*Page {page_idx}*\n")
        if page_lines:
            parts.append("\n\n".join(page_lines))

    doc.close()
    return "\n\n".join(parts) + "\n"


def ingest_directory(pdf_dir: Path, output_dir: Path, start_doc_id: int = 90001):
    """
    Convert all PDFs in pdf_dir named like:
      {companyID}_{YEAR}_{optional_slug}.pdf
    into:
      output_dir/{docID}_{companyID}_{YEAR}.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    manifest = []
    next_id = start_doc_id

    for pdf_file in pdf_files:
        m = re.match(r"^(\d+)_(202[56])(?:_.*)?\.pdf$", pdf_file.name, re.IGNORECASE)
        if not m:
            print(
                f"[SKIP] {pdf_file.name} does not match '{{companyID}}_{{2025|2026}}_*.pdf'"
            )
            continue

        company_id = int(m.group(1))
        year = int(m.group(2))
        doc_id = next_id
        next_id += 1

        out_name = f"{doc_id}_{company_id}_{year}.md"
        out_path = output_dir / out_name

        try:
            md_text = pdf_to_corpus_markdown(pdf_file, doc_id)
            if len(md_text.strip()) < 500:
                print(f"[WARN] {pdf_file.name} produced very little text (<500 chars), skipping.")
                continue
            out_path.write_text(md_text, encoding="utf-8")
            manifest.append(
                {
                    "doc_id": doc_id,
                    "company_id": company_id,
                    "year": year,
                    "source_pdf": pdf_file.name,
                    "output_md": out_name,
                    "size_bytes": out_path.stat().st_size,
                }
            )
            print(f"[OK] {pdf_file.name} -> {out_name} ({out_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"[ERR] Failed to convert {pdf_file.name}: {e}")

    manifest_path = Path("outputs/new_2025_2026_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nIngested {len(manifest)} new documents. Manifest saved to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest 2025-2026 ESG PDFs into data/raw/")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/raw_pdfs_2025_2026"),
        help="Directory containing {companyID}_{YEAR}_{name}.pdf files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target directory for converted .md files",
    )
    parser.add_argument(
        "--start-doc-id",
        type=int,
        default=90001,
        help="Starting Document ID for newly ingested files",
    )
    args = parser.parse_args()
    ingest_directory(args.pdf_dir, args.output_dir, args.start_doc_id)

