import threading

import psycopg2
import select


class PostgresFilters:
    """Class to manage PostgreSQL filters with real-time updates."""

    # --- Constants ---
    HOST = "localhost"
    PORT = 5432
    DBNAME = "filters"
    USER = "munir"
    PASSWORD = "123"

    def __init__(self):
        self.conn = psycopg2.connect(
            host=self.HOST,
            port=self.PORT,
            dbname=self.DBNAME,
            user=self.USER,
            password=self.PASSWORD
        )
        self.conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.all_authors = []
        self.all_sources = []

        self.update_filters()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def listen(self) -> None:
        """Listen for changes in the filters table and update authors and sources accordingly."""
        cur = self.conn.cursor()
        cur.execute("LISTEN filter_changes;")

        while True:
            if select.select([self.conn], [], [], 1) == ([], [], []):
                continue

            self.conn.poll()

            while self.conn.notifies:
                _ = self.conn.notifies.pop(0)
                self.update_filters()

        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()

    def update_filters(self) -> None:
        """Update the list of authors and sources from the database."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT authors FROM filters;")
        self.all_authors = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT DISTINCT sources FROM filters;")
        self.all_sources = [row[0] for row in cur.fetchall()]

    def __exit__(self):
        """Exit the runtime context related to this object."""
        self.conn.close()