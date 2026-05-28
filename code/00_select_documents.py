# %%
import os
import random
import pandas as pd
import json
from pathlib import Path
import re
from collections import defaultdict

def extract_year_from_filename(filename):
    """Extract year from filename like '102_15_2013.md'"""
    match = re.search(r'_(\d{4})\.md$', filename)
    return int(match.group(1)) if match else None

def analyze_document_corpus():
    """Analyze the document corpus to understand distribution"""
    docs_dir = Path("data/raw")
    documents = []
    
    for file_path in docs_dir.glob("*.md"):
        # Extract year from filename
        year = extract_year_from_filename(file_path.name)
        
        # Get file size as proxy for document length
        file_size = file_path.stat().st_size
        
        documents.append({
            'filename': file_path.name,
            'filepath': str(file_path),
            'year': year,
            'size': file_size
        })
    
    df = pd.DataFrame(documents)
    
    # Analyze distribution
    print("=== Document Corpus Analysis ===")
    print(f"Total documents: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Documents per year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} documents")
    
    # Size distribution
    print(f"\nDocument size distribution:")
    print(f"  Mean: {df['size'].mean():.0f} bytes")
    print(f"  Median: {df['size'].median():.0f} bytes")
    print(f"  Min: {df['size'].min()} bytes")
    print(f"  Max: {df['size'].max()} bytes")
    
    return df

def select_all_documents(df):
    """Select all documents for full dataset evaluation"""
    
    # Return all documents sorted by year and size for consistency
    selected_docs = df.sort_values(['year', 'size']).to_dict('records')
    
    return selected_docs

def create_document_dataset(selected_docs):
    """Create document dataset for experiments"""
    
    # Create output directory
    output_dir = Path("data/documents")
    output_dir.mkdir(exist_ok=True)
    
    # Copy selected documents
    selected_filenames = []
    selection_info = []
    
    for doc in selected_docs:
        # Copy document
        src_path = Path(doc['filepath'])
        dst_path = output_dir / doc['filename']
        
        # Read and write content
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        selected_filenames.append(doc['filename'])
        selection_info.append({
            'filename': doc['filename'],
            'year': doc['year'],
            'size': doc['size']
        })
    
    # Save selection info
    selection_df = pd.DataFrame(selection_info)
    selection_df.to_csv("outputs/document_selection_info.csv", index=False)
    
    # Save as JSON for reference
    with open("outputs/selected_documents.json", "w") as f:
        json.dump(selection_info, f, indent=2)
    
    print(f"\n=== Document Selection Complete ===")
    print(f"Selected {len(selected_docs)} documents")
    print(f"Documents saved to: {output_dir}")
    print(f"Selection info saved to: outputs/document_selection_info.csv")
    
    # Print selection summary
    print(f"\nSelection summary by year:")
    year_summary = selection_df['year'].value_counts().sort_index()
    for year, count in year_summary.items():
        print(f"  {year}: {count} documents")
    
    print(f"\nSize distribution:")
    print(f"  Mean: {selection_df['size'].mean():.0f} bytes")
    print(f"  Median: {selection_df['size'].median():.0f} bytes")
    print(f"  Min: {selection_df['size'].min()} bytes")
    print(f"  Max: {selection_df['size'].max()} bytes")
    
    return output_dir

def main():
    """Main function to select all documents for full dataset evaluation"""
    print("Starting full document selection for ESG RAG experiments...")
    
    # Analyze corpus
    df = analyze_document_corpus()
    
    # Select all documents
    selected_docs = select_all_documents(df)
    
    # Create dataset
    dataset_dir = create_document_dataset(selected_docs)
    
    print(f"\nDocument selection completed!")
    print(f"Ready to use {dataset_dir} for experiments")

if __name__ == "__main__":
    main()
