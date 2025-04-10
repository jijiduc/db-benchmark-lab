#!/usr/bin/env python3
"""
Module to interact with PostgreSQL for the mouse tracking benchmark.
"""
import psycopg2
import psycopg2.extras
import json
import time
from datetime import datetime

class PostgreSQLClient:
    """Client for interacting with PostgreSQL."""
    
    def __init__(self, connection_params="dbname=mousebenchmark user=jduc password=password", table_name="mouse_events"):
        """Initialize the connection to PostgreSQL."""
        # Replace YOUR_USERNAME with your actual username
        self.connection_params = connection_params
        self.table_name = table_name
        self.conn = psycopg2.connect(self.connection_params)
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        self._ensure_table_exists()
        
    def _ensure_table_exists(self):
        """Create the table if it doesn't exist."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_id ON {self.table_name} (user_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data_user_id ON {self.table_name} ((data->>'user_id'));
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data_gin ON {self.table_name} USING GIN (data);
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()
        
    def insert_event(self, event):
        """Insert a mouse event in the table."""
        user_id = event.get('user_id')
        query = f"""
        INSERT INTO {self.table_name} (user_id, data)
        VALUES (%s, %s)
        RETURNING id
        """
        self.cursor.execute(query, (user_id, json.dumps(event)))
        result = self.cursor.fetchone()
        self.conn.commit()
        return result[0] if result else None
        
    def insert_events(self, events):
        """Insert multiple mouse events in the table."""
        values = [(event.get('user_id'), json.dumps(event)) for event in events]
        args_str = ','.join(self.cursor.mogrify("(%s,%s)", v).decode('utf-8') for v in values)
        
        query = f"""
        INSERT INTO {self.table_name} (user_id, data)
        VALUES {args_str}
        RETURNING id
        """
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        self.conn.commit()
        return [r[0] for r in result]
        
    def find_events_by_user(self, user_id):
        """Get all events from a user."""
        query = f"""
        SELECT id, user_id, data, created_at FROM {self.table_name}
        WHERE user_id = %s
        ORDER BY created_at DESC
        """
        self.cursor.execute(query, (user_id,))
        return self.cursor.fetchall()
        
    def find_events_by_page(self, page):
        """Get all events from a specific page."""
        query = f"""
        SELECT id, user_id, data, created_at FROM {self.table_name}
        WHERE data->>'page' = %s
        ORDER BY created_at DESC
        """
        self.cursor.execute(query, (page,))
        return self.cursor.fetchall()
        
    def count_events(self):
        """Count the total number of events."""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]
        
    def clear_table(self):
        """Clear the events table."""
        query = f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY"
        self.cursor.execute(query)
        self.conn.commit()
        
    def close(self):
        """Close the connection to PostgreSQL."""
        self.cursor.close()
        self.conn.close()
        
    def __enter__(self):
        """Support for context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up."""
        self.close()