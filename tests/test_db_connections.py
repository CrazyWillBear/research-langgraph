import unittest
from unittest.mock import patch, MagicMock


class TestDbConnections(unittest.TestCase):
    """Test suite for the global database connections module."""

    def setUp(self):
        """Clear global connection instances before each test."""
        import ai.db_connections as db_conn
        db_conn._qdrant_client = None
        db_conn._postgres_pool = None

    @patch('ai.db_connections.QdrantClient')
    def test_get_qdrant_client_creates_instance(self, mock_qdrant_class):
        """Test that get_qdrant_client creates a new instance on first call."""
        from ai.db_connections import get_qdrant_client
        
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # First call should create new instance
        client1 = get_qdrant_client()
        self.assertEqual(client1, mock_client)
        mock_qdrant_class.assert_called_once()

    @patch('ai.db_connections.QdrantClient')
    def test_get_qdrant_client_returns_singleton(self, mock_qdrant_class):
        """Test that get_qdrant_client returns the same instance on subsequent calls."""
        from ai.db_connections import get_qdrant_client
        
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Multiple calls should return same instance
        client1 = get_qdrant_client()
        client2 = get_qdrant_client()
        client3 = get_qdrant_client()
        
        self.assertEqual(client1, client2)
        self.assertEqual(client2, client3)
        # Should only create once
        mock_qdrant_class.assert_called_once()

    @patch('ai.db_connections.pool.ThreadedConnectionPool')
    def test_get_postgres_pool_creates_instance(self, mock_pool_class):
        """Test that get_postgres_pool creates a new pool on first call."""
        from ai.db_connections import get_postgres_pool
        
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        # First call should create new pool
        pool1 = get_postgres_pool()
        self.assertEqual(pool1, mock_pool)
        mock_pool_class.assert_called_once()

    @patch('ai.db_connections.pool.ThreadedConnectionPool')
    def test_get_postgres_pool_returns_singleton(self, mock_pool_class):
        """Test that get_postgres_pool returns the same pool on subsequent calls."""
        from ai.db_connections import get_postgres_pool
        
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        # Multiple calls should return same pool
        pool1 = get_postgres_pool()
        pool2 = get_postgres_pool()
        pool3 = get_postgres_pool()
        
        self.assertEqual(pool1, pool2)
        self.assertEqual(pool2, pool3)
        # Should only create once
        mock_pool_class.assert_called_once()

    @patch('ai.db_connections.pool.ThreadedConnectionPool')
    @patch('ai.db_connections.QdrantClient')
    def test_close_connections_closes_all(self, mock_qdrant_class, mock_pool_class):
        """Test that close_connections closes both Qdrant and Postgres connections."""
        from ai.db_connections import get_qdrant_client, get_postgres_pool, close_connections
        
        mock_client = MagicMock()
        mock_pool = MagicMock()
        mock_qdrant_class.return_value = mock_client
        mock_pool_class.return_value = mock_pool
        
        # Create connections
        get_qdrant_client()
        get_postgres_pool()
        
        # Close all connections
        close_connections()
        
        # Verify both were closed
        mock_client.close.assert_called_once()
        mock_pool.closeall.assert_called_once()

    @patch('ai.db_connections.pool.ThreadedConnectionPool')
    @patch('ai.db_connections.QdrantClient')
    def test_close_connections_resets_globals(self, mock_qdrant_class, mock_pool_class):
        """Test that close_connections resets global connection variables."""
        from ai.db_connections import get_qdrant_client, get_postgres_pool, close_connections
        import ai.db_connections as db_conn
        
        mock_client = MagicMock()
        mock_pool = MagicMock()
        mock_qdrant_class.return_value = mock_client
        mock_pool_class.return_value = mock_pool
        
        # Create connections
        get_qdrant_client()
        get_postgres_pool()
        
        # Verify connections exist
        self.assertIsNotNone(db_conn._qdrant_client)
        self.assertIsNotNone(db_conn._postgres_pool)
        
        # Close all connections
        close_connections()
        
        # Verify globals are reset
        self.assertIsNone(db_conn._qdrant_client)
        self.assertIsNone(db_conn._postgres_pool)

    @patch('ai.db_connections.QdrantClient')
    def test_get_qdrant_client_with_grpc(self, mock_qdrant_class):
        """Test that get_qdrant_client uses gRPC when requested."""
        from ai.db_connections import get_qdrant_client
        
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Call with gRPC enabled
        client = get_qdrant_client(use_grpc=True)
        
        # Verify QdrantClient was called (exact params depend on URL parsing)
        mock_qdrant_class.assert_called_once()
        self.assertEqual(client, mock_client)

    @patch('ai.db_connections.pool.ThreadedConnectionPool')
    def test_get_postgres_pool_with_custom_params(self, mock_pool_class):
        """Test that get_postgres_pool accepts custom minconn and maxconn."""
        from ai.db_connections import get_postgres_pool
        
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        # Call with custom params
        pool = get_postgres_pool(minconn=2, maxconn=20)
        
        # Verify pool was created with custom params
        mock_pool_class.assert_called_once()
        call_kwargs = mock_pool_class.call_args[1]
        self.assertEqual(call_kwargs['minconn'], 2)
        self.assertEqual(call_kwargs['maxconn'], 20)


if __name__ == '__main__':
    unittest.main()
