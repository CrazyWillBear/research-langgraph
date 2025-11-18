import unittest
from unittest.mock import patch, MagicMock, call

from ai.research_agent.nodes.query_vector_db import query_vector_db


class TestQueryVectorDb(unittest.TestCase):
    """Test suite for the query_vector_db node function."""

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_basic(self, mock_get_qdrant, mock_get_postgres, mock_embed):
        """Test basic vector database query without filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1', 'result2']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state
        state = {
            'queries': [
                MagicMock(query='test query', filters=None)
            ],
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        self.assertEqual(len(result['resources']), 2)
        mock_embed.assert_called_once_with('test query')
        mock_qdrant_instance.query_points.assert_called_once()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_with_author_filter(self, mock_get_qdrant, mock_get_postgres, mock_extract, mock_embed):
        """Test vector database query with author filter."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.return_value = ('Aristotle', 95)  # (match, score)

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Aristotle',), ('Plato',), ('Kant',)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state with author filter
        filters_obj = MagicMock()
        filters_obj.author = 'Aristotle'
        filters_obj.source_title = None
        state = {
            'queries': [
                MagicMock(query='virtue ethics', filters=filters_obj)
            ],
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        mock_cur.execute.assert_called()
        mock_extract.assert_called_once()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_with_source_filter(self, mock_get_qdrant, mock_get_postgres, mock_extract, mock_embed):
        """Test vector database query with source_title filter."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.return_value = ('Republic', 90)

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Republic',), ('Ethics',)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state with source filter
        filters_obj = MagicMock()
        filters_obj.author = None
        filters_obj.source_title = 'Republic'
        state = {
            'queries': [
                MagicMock(query='theory of forms', filters=filters_obj)
            ],
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_with_both_filters(self, mock_get_qdrant, mock_get_postgres, mock_extract, mock_embed):
        """Test vector database query with both author and source_title filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.side_effect = [('Kant', 95), ('Groundwork', 90)]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = [
            [('Kant',), ('Hume',)],  # authors
            [('Groundwork',), ('Critique',)]  # sources
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state with both filters
        filters_obj = MagicMock()
        filters_obj.author = 'Kant'
        filters_obj.source_title = 'Groundwork'
        state = {
            'queries': [
                MagicMock(query='categorical imperative', filters=filters_obj)
            ],
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        self.assertEqual(mock_extract.call_count, 2)

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_accumulates_resources(self, mock_get_qdrant, mock_get_postgres, mock_embed):
        """Test that new resources are accumulated with existing ones."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['new_result1', 'new_result2']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state with existing resources
        state = {
            'queries': [
                MagicMock(query='test query', filters=None)
            ],
            'resources': ['old_result1', 'old_result2']
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions - should have 4 total resources (2 old + 2 new)
        self.assertEqual(len(result['resources']), 4)
        self.assertIn('old_result1', result['resources'])
        self.assertIn('new_result1', result['resources'])

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_closes_connections(self, mock_get_qdrant, mock_get_postgres, mock_embed):
        """Test that database connections are properly returned to pool."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state
        state = {
            'queries': [
                MagicMock(query='test', filters=None)
            ],
            'resources': []
        }

        # Call query_vector_db
        query_vector_db(state)

        # Assertions - connection should be returned to pool (not closed)
        mock_pool.putconn.assert_called_once_with(mock_conn)
        # Global connections should not be closed
        mock_qdrant_instance.close.assert_not_called()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.get_postgres_pool')
    @patch('ai.research_agent.nodes.query_vector_db.get_qdrant_client')
    def test_query_vector_db_fuzzy_matching(self, mock_get_qdrant, mock_get_postgres, mock_extract, mock_embed):
        """Test that fuzzy matching is used for filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        # Simulate fuzzy match finding close match
        mock_extract.return_value = ('Aristoteles', 85)  # Close but not exact

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_get_qdrant.return_value = mock_qdrant_instance

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Aristoteles',), ('Plato',)]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.getconn.return_value = mock_conn
        mock_get_postgres.return_value = mock_pool

        # Create state with slightly misspelled author
        filters_obj = MagicMock()
        filters_obj.author = 'Aristotle'
        filters_obj.source_title = None
        state = {
            'queries': [
                MagicMock(query='ethics', filters=filters_obj)
            ],
            'resources': []
        }

        # Call query_vector_db
        query_vector_db(state)

        # Verify fuzzy matching was used
        mock_extract.assert_called_once()
        # First arg should be the query, second should be list of candidates
        args = mock_extract.call_args[0]
        self.assertEqual(args[0], 'Aristotle')


if __name__ == '__main__':
    unittest.main()

