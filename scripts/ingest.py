import asyncio
import json
import logging
import sys
import argparse

from xenrag.retrieval.stores.qdrant import QdrantVectorStore
from xenrag.retrieval.stores.neo4j import Neo4jGraphStore
from xenrag.ingestion.pipeline import (
    process_batch,
    process_for_vector_store,
    process_for_graph_store
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest")

async def ingest_file(file_path: str, limit: int = None, batch_size: int = 50):
    """
    Ingest data from JSON or JSONL file using the enhanced pipeline.
    """
    logger.info(f"Reading file: {file_path}")
    
    vector_store = QdrantVectorStore()
    graph_store = Neo4jGraphStore()
    
    batch = []
    total_documents = 0
    total_segments = 0
    
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
            batch.append(item)
            
            if len(batch) >= batch_size:
                segments = await process_and_store(vector_store, graph_store, batch)
                total_documents += len(batch)
                total_segments += segments
                logger.info(f"Processed {total_documents} docs -> {total_segments} segments")
                batch = []

        if batch:
            segments = await process_and_store(vector_store, graph_store, batch)
            total_documents += len(batch)
            total_segments += segments
            
        logger.info(f"Ingestion Complete!")
        logger.info(f"  Documents: {total_documents}")
        logger.info(f"  Segments: {total_segments}")
        logger.info(f"  Avg segments/doc: {total_segments/max(total_documents,1):.1f}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        graph_store.close()


async def process_and_store(vector_store, graph_store, batch) -> int:
    """Process batch through pipeline and store in both stores."""
    
    # Run through pipeline
    segments = process_batch(batch)
    
    # Prepare for each store
    vector_docs = process_for_vector_store(segments)
    graph_docs = process_for_graph_store(segments)
    
    # Store in Qdrant
    if vector_docs:
        vector_store.add_documents(vector_docs)
    
    # Store in Neo4j
    if graph_docs:
        graph_store.add_documents(graph_docs, label="ReviewSegment")
    
    return len(segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into XenRAG with enhanced pipeline")
    parser.add_argument("file", help="Path to JSON or JSONL file")
    parser.add_argument("--limit", type=int, help="Limit number of documents", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size", default=50)
    
    args = parser.parse_args()
    asyncio.run(ingest_file(args.file, args.limit, args.batch_size))
