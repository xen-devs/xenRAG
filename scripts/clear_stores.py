"""
Script to clear all data from Qdrant and Neo4j stores.
"""

import logging
from xenrag.config import settings
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("clear_stores")


def clear_qdrant():
    """Clear all data from Qdrant collection."""
    logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
    
    client = QdrantClient(url=settings.QDRANT_URL)
    collection_name = settings.QDRANT_COLLECTION
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            # Get count before deletion
            info = client.get_collection(collection_name)
            count = info.points_count
            
            # Delete and recreate collection
            client.delete_collection(collection_name)
            logger.info(f"Deleted Qdrant collection '{collection_name}' ({count} points)")
        else:
            logger.info(f"Qdrant collection '{collection_name}' does not exist")
            
    except Exception as e:
        logger.error(f"Failed to clear Qdrant: {e}")


def clear_neo4j():
    """Clear all nodes and relationships from Neo4j."""
    logger.info(f"Connecting to Neo4j at {settings.NEO4J_URI}")
    
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    
    try:
        with driver.session() as session:
            # Get count before deletion
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info(f"Deleted all Neo4j nodes ({count} nodes)")
            
    except Exception as e:
        logger.error(f"Failed to clear Neo4j: {e}")
    finally:
        driver.close()


def main():
    logger.info("Clearing all data stores...")
    
    clear_qdrant()
    clear_neo4j()
    
    logger.info("Done! Ready for fresh ingestion.")


if __name__ == "__main__":
    main()
