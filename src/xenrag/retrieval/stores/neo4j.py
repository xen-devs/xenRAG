import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from xenrag.retrieval.interfaces import GraphStore
from xenrag.retrieval.types import RetrievalItem, SearchContext
from xenrag.config import settings

logger = logging.getLogger(__name__)

class Neo4jGraphStore(GraphStore):
    """
    Neo4j Adapter
    """
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        
        logger.info(f"Connecting to Neo4j at {self.uri}")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified.")
        except Exception as e:
            logger.critical(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")

    def query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[RetrievalItem]:
        """Execute a raw cypher query."""
        logger.debug(f"Executing Cypher: {cypher_query[:50]}...")
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})
                items = []
                for record in result:
                    # Naive serialization of record to string for now
                    content = str(record.data())
                    items.append(RetrievalItem(
                        id="neo4j_res",
                        content=content,
                        source="neo4j",
                        score=1.0, 
                        metadata=record.data()
                    ))
                logger.info(f"Neo4j query returned {len(items)} records.")
                return items
        except neo4j_exceptions.Neo4jError as e:
            logger.error(f"Neo4j query failed: {e.message}")
            return []
        except Exception as e:
            logger.error(f"Unexpected Neo4j error: {e}")
            return []
    
    def get_context(self, context: SearchContext) -> List[RetrievalItem]:
        """
        Retrieve relevant graph context. 
        """
        cypher = """
        MATCH (n)
        WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($query))
        RETURN n, labels(n) as labels LIMIT $limit
        """
        return self.query(cypher, {"query": context.query, "limit": context.limit})

    def add_documents(self, documents: List[Dict[str, Any]], label: str = "Document") -> None:
        """
        Ingest documents as nodes.
        Uses UNWIND for efficient batch insertion.
        """
        logger.info(f"Ingesting {len(documents)} nodes into Graph [Label: {label}]")
        
        cypher = f"""
        UNWIND $batch AS row
        MERGE (n:`{label}` {{id: row.id}})
        ON CREATE SET n = row
        ON MATCH SET n += row
        """
        
        import uuid
        prepared_batch = []
        for doc in documents:
            d = doc.copy()
            if "id" not in d:
                d["id"] = str(uuid.uuid4())
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    d[k] = str(v)
            prepared_batch.append(d)

        try:
            with self.driver.session() as session:
                session.run(cypher, {"batch": prepared_batch})
            logger.info("Graph ingestion complete.")
        except Exception as e:
            logger.error(f"Graph ingestion failed: {e}")
            raise
