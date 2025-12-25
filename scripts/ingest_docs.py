#!/usr/bin/env python3
"""
Data ingestion script for the Intelligence Layer - RAG Backend
Recursively parses .md and .mdx files from the docs/ folder, chunks content by sub-chapters/headers,
generates embeddings via Cohere, and upserts them to Qdrant with metadata (title, module, chapter_url)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import uuid
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the server directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.utils.text_splitter import TextSplitter
from server.utils.validators import validate_file_type, validate_document_chunk
from server.services.embedding import get_embedding
from server.services.vector_store import vector_store_service
from server.utils.logging import get_logger

# Create a default text splitter instance
default_text_splitter = TextSplitter()

# Set up logging
logger = get_logger(__name__)

def get_all_files(source_path: str, file_types: List[str], recursive: bool = True) -> List[Path]:
    """
    Get all files of specified types from source path
    """
    source = Path(source_path)
    files = []

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source.is_file():
        if source.suffix.lower() in file_types:
            files.append(source)
    else:
        if recursive:
            # Recursively find all files with specified extensions
            for file_type in file_types:
                files.extend(source.rglob(f"*{file_type}"))
        else:
            # Only find files in the top-level directory
            for file_type in file_types:
                files.extend(source.glob(f"*{file_type}"))

    return files

def read_file_content(file_path: Path) -> str:
    """
    Read content from a file, handling both .md and .mdx files
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path} with alternative encoding: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def process_file(file_path: Path, text_splitter: TextSplitter) -> List[Dict[str, Any]]:
    """
    Process a single file: read content, split into chunks, and prepare for ingestion
    """
    try:
        logger.info(f"Processing file: {file_path}")

        # Read file content
        content = read_file_content(file_path)

        # Split content into chunks
        chunks = text_splitter.split_by_headers(content, str(file_path))

        # Prepare chunks for ingestion
        processed_chunks = []
        for chunk in chunks:
            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}#{chunk['title']}"))

            # Prepare the document for ingestion
            document = {
                "id": chunk_id,
                "title": chunk["title"],
                "module": chunk["module"],
                "chapter_url": chunk["chapter_url"],
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }

            # Validate the document
            try:
                validate_document_chunk(document)
                processed_chunks.append(document)
            except Exception as e:
                logger.warning(f"Skipping chunk due to validation error in {file_path}: {str(e)}")
                continue

        logger.info(f"Processed {len(processed_chunks)} chunks from {file_path}")
        return processed_chunks

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def generate_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of document chunks
    """
    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Extract content for batch processing
        contents = [chunk["content"] for chunk in chunks]

        # Generate embeddings one at a time with conservative rate limiting to avoid hitting limits
        embeddings = []

        import time

        for i, content in enumerate(contents):
            # Try single embedding at a time to be more conservative
            retry_count = 0
            max_retries = 3
            embedding_success = False

            while retry_count < max_retries and not embedding_success:
                try:
                    batch_embeddings = get_embedding([content], input_type="search_document")
                    embeddings.extend(batch_embeddings)
                    embedding_success = True
                    logger.debug(f"Successfully generated embedding for chunk {i+1}/{len(contents)}")
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        retry_count += 1
                        wait_time = 15 * retry_count  # Exponential backoff starting at 15 seconds
                        logger.warning(f"Rate limit hit on chunk {i+1}, waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Error generating embedding for chunk {i+1}: {str(e)}")
                        break

            if not embedding_success:
                logger.error(f"Failed to generate embedding for chunk {i+1} after {max_retries} retries")
                embeddings.append([])  # Placeholder for failed embedding

            # Add conservative delay between each request to avoid rate limits
            time.sleep(3)

        # Check if any embeddings failed and filter them out
        valid_embeddings = []
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if embeddings[i] and len(embeddings[i]) > 0:  # Only add chunks with valid embeddings
                chunk["embedding"] = embeddings[i]
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk due to failed embedding: {chunk['title']}")

        logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
        return valid_chunks

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def ingest_file(file_path: Path, text_splitter: TextSplitter) -> int:
    """
    Process a single file and ingest it into the vector store
    Returns the number of chunks successfully ingested
    """
    try:
        # Process the file to get document chunks
        chunks = process_file(file_path, text_splitter)

        if not chunks:
            logger.warning(f"No valid chunks found in {file_path}")
            return 0

        # Generate embeddings for the chunks
        chunks_with_embeddings = generate_embeddings_for_chunks(chunks)

        # Prepare documents for upsert to Qdrant
        documents_for_upsert = []
        for chunk in chunks_with_embeddings:
            document = {
                "id": chunk["id"],
                "title": chunk["title"],
                "module": chunk["module"],
                "chapter_url": chunk["chapter_url"],
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }
            documents_for_upsert.append(document)

        # Upsert documents to Qdrant with retry logic
        import time
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                success = vector_store_service.upsert_documents(documents_for_upsert)
            except Exception as upsert_error:
                if "timeout" in str(upsert_error).lower() or "rate" in str(upsert_error).lower():
                    retry_count += 1
                    logger.warning(f"Upsert failed due to rate limit or timeout, retry {retry_count}/{max_retries}")
                    time.sleep(5 * retry_count)  # Exponential backoff
                else:
                    logger.error(f"Upsert failed with non-retryable error: {str(upsert_error)}")
                    break

        if success:
            logger.info(f"Successfully ingested {len(documents_for_upsert)} chunks from {file_path}")
            return len(documents_for_upsert)
        else:
            logger.error(f"Failed to ingest chunks from {file_path}")
            return 0

    except Exception as e:
        logger.error(f"Error ingesting file {file_path}: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Ingest documentation files into the RAG system")
    parser.add_argument(
        "--source-path",
        type=str,
        default="docs",
        help="Path to the documentation source files to ingest (default: docs)"
    )
    parser.add_argument(
        "--recursive",
        type=bool,
        default=True,
        help="Whether to recursively scan subdirectories (default: True)"
    )
    parser.add_argument(
        "--file-types",
        type=str,
        nargs="+",
        default=[".md", ".mdx"],
        help="File types to process (default: .md .mdx)"
    )

    args = parser.parse_args()

    # Validate arguments
    source_path = args.source_path
    recursive = args.recursive
    file_types = args.file_types

    logger.info(f"Starting ingestion process...")
    logger.info(f"Source path: {source_path}")
    logger.info(f"Recursive: {recursive}")
    logger.info(f"File types: {file_types}")

    try:
        # Get all files to process
        files = get_all_files(source_path, file_types, recursive)
        logger.info(f"Found {len(files)} files to process")

        if not files:
            logger.warning("No files found to process")
            return

        # Process each file
        total_processed_files = 0
        total_indexed_chunks = 0
        errors = []

        for file_path in files:
            try:
                chunks_ingested = ingest_file(file_path, default_text_splitter)
                if chunks_ingested > 0:
                    total_processed_files += 1
                    total_indexed_chunks += chunks_ingested
                    logger.info(f"Successfully processed {file_path}")
                else:
                    errors.append(f"No chunks ingested from {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Print summary
        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Files processed: {total_processed_files}/{len(files)}")
        print(f"Chunks indexed: {total_indexed_chunks}")
        print(f"Errors: {len(errors)}")

        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(f"  - {error}")

        print("="*50)

        # Print final status
        if total_processed_files > 0:
            status = "success"
        elif len(errors) == len(files):
            status = "failed"
        else:
            status = "partial"

        print(f"Overall status: {status}")
        print("="*50)

    except Exception as e:
        logger.error(f"Critical error during ingestion: {str(e)}")
        print(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()