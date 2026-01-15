import asyncio
import json
import logging
import sys

from xenrag.retrieval.stores.qdrant import QdrantVectorStore
from xenrag.retrieval.stores.neo4j import Neo4jGraphStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingest")

async def ingest_file(file_path: str, limit: int = None):
    """
    Ingest data from JSON or JSONL file in batches.
    """
    logger.info(f"Reading file: {file_path}")
    
    vector_store = QdrantVectorStore()
    graph_store = Neo4jGraphStore()
    
    BATCH_SIZE = 50
    batch = []
    total_processed = 0
    
    try:
        # Determine file type and generator
        def item_generator():
            count = 0
            is_jsonl = file_path.endswith('.jsonl')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if is_jsonl:
                    for line in f:
                        if not line.strip(): continue
                        yield json.loads(line)
                        count += 1
                        if limit and count >= limit: break
                else:
                    # Standard JSON list
                    data = json.load(f)
                    if not isinstance(data, list): data = [data]
                    for item in data:
                        yield item
                        count += 1
                        if limit and count >= limit: break

        # 2. Process Loop
        for item in item_generator():
            # Heuristic for text
            text_content = item.get("text") or item.get("content") or item.get("review") or item.get("body")
            
            if not text_content:
                text_content = json.dumps(item)
            
            # Prepare item
            processed_item = item.copy()
            processed_item["text"] = text_content
            
            # Add to batch
            batch.append(processed_item)
            
            # Flush batch if full
            if len(batch) >= BATCH_SIZE:
                await process_batch(vector_store, graph_store, batch)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed} items...")
                batch = []

        # Flush remaining
        if batch:
            await process_batch(vector_store, graph_store, batch)
            total_processed += len(batch)
            
        logger.info(f"Ingestion Complete! Total documents: {total_processed}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'graph_store' in locals():
            graph_store.close()

async def process_batch(vector_store, graph_store, batch):
    """Result of processing a single batch."""
    # Qdrant
    vector_store.add_documents(batch)
    # Neo4j
    graph_store.add_documents(batch, label="Review")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest data into XenRAG")
    parser.add_argument("file", help="Path to JSON or JSONL file")
    parser.add_argument("--limit", type=int, help="Limit number of documents to ingest", default=None)
    
    args = parser.parse_args()
    asyncio.run(ingest_file(args.file, args.limit))
