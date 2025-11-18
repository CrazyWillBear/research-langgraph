"""
Global database connections module.
Provides singleton QdrantClient with gRPC support and Postgres connection pool.
"""
from qdrant_client import QdrantClient
from psycopg2 import pool
import os

# --- Constants ---
VEC_DB_URL = os.getenv("VEC_DB_URL", "http://localhost:6333")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "filters")
POSTGRES_USER = os.getenv("POSTGRES_USER", "munir")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "123")

# --- Global connection instances (initialized lazily) ---
_qdrant_client = None
_postgres_pool = None


def get_qdrant_client(use_grpc: bool = True) -> QdrantClient:
    """
    Get or create the global QdrantClient instance.
    
    Args:
        use_grpc: Whether to use gRPC protocol (default: True for better performance)
    
    Returns:
        QdrantClient: The global Qdrant client instance
    """
    global _qdrant_client
    
    if _qdrant_client is None:
        # Parse URL to determine if we should use gRPC
        if use_grpc and VEC_DB_URL.startswith("http://"):
            # Convert HTTP URL to gRPC format (host:port)
            # e.g., http://localhost:6333 -> localhost:6334 (gRPC default port)
            host = VEC_DB_URL.replace("http://", "").replace("https://", "").split(":")[0]
            grpc_port = 6334  # Default gRPC port for Qdrant
            _qdrant_client = QdrantClient(host=host, port=grpc_port, prefer_grpc=True)
        else:
            # Use HTTP connection
            _qdrant_client = QdrantClient(url=VEC_DB_URL)
    
    return _qdrant_client


def get_postgres_pool(minconn: int = 1, maxconn: int = 10):
    """
    Get or create the global Postgres connection pool.
    
    Args:
        minconn: Minimum number of connections in the pool
        maxconn: Maximum number of connections in the pool
    
    Returns:
        psycopg2.pool.ThreadedConnectionPool: The global Postgres connection pool
    """
    global _postgres_pool
    
    if _postgres_pool is None:
        _postgres_pool = pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
    
    return _postgres_pool


def close_connections():
    """
    Close all global database connections.
    Should be called when the application is shutting down.
    """
    global _qdrant_client, _postgres_pool
    
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None
    
    if _postgres_pool is not None:
        _postgres_pool.closeall()
        _postgres_pool = None
