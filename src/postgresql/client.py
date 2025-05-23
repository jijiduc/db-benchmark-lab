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
    
    # 1. Advanced queries
    def find_events_by_time_range(self, start_date, end_date):
        """Find events within a specific time range."""
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        query = """
        SELECT id, user_id, data, created_at FROM mouse_events 
        WHERE (data->>'timestamp')::timestamp >= %s::timestamp 
        AND (data->>'timestamp')::timestamp <= %s::timestamp
        """
        
        self.cursor.execute(query, (start_str, end_str))
        results = self.cursor.fetchall()
        return list(results)

    def find_events_in_screen_zone(self, x_min, x_max, y_min, y_max):
        """Find events within a specific screen area."""
        query = """
        SELECT id, user_id, data, created_at FROM mouse_events 
        WHERE (data->>'x_pos')::float >= %s 
        AND (data->>'x_pos')::float <= %s
        AND (data->>'y_pos')::float >= %s 
        AND (data->>'y_pos')::float <= %s
        """
        
        self.cursor.execute(query, (x_min, x_max, y_min, y_max))
        results = self.cursor.fetchall()
        return list(results)

    def aggregate_events_by_user(self):
        """Aggregate events by user."""
        query = """
        SELECT data->>'user_id' as user_id, COUNT(*) as count
        FROM mouse_events
        GROUP BY data->>'user_id'
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # For each user, fetch their events
        aggregated_results = []
        for row in results:
            user_id = row[0]  # First column is user_id
            count = row[1]    # Second column is count
            
            # Now get all events for this user
            self.cursor.execute("""
                SELECT id, user_id, data, created_at FROM mouse_events
                WHERE data->>'user_id' = %s
            """, (user_id,))
            events = self.cursor.fetchall()
            
            aggregated_results.append({
                'user_id': user_id,
                'count': count,
                'events': events
            })
        
        return aggregated_results

    def aggregate_events_by_page(self):
        """Aggregate events by page."""
        query = """
        SELECT data->>'page' as page, COUNT(*) as count
        FROM mouse_events
        GROUP BY data->>'page'
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # For each page, fetch related events
        aggregated_results = []
        for row in results:
            page = row[0]    # First column is page
            count = row[1]   # Second column is count
            
            # Now get all events for this page
            self.cursor.execute("""
                SELECT id, user_id, data, created_at FROM mouse_events
                WHERE data->>'page' = %s
            """, (page,))
            events = self.cursor.fetchall()
            
            aggregated_results.append({
                'page': page,
                'count': count,
                'events': events
            })
        
        return aggregated_results

    def aggregate_events_by_device(self):
        """Aggregate events by device type."""
        query = """
        SELECT data->>'device' as device, COUNT(*) as count
        FROM mouse_events
        GROUP BY data->>'device'
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # For each device, fetch related events
        aggregated_results = []
        for row in results:
            device = row[0]    # First column is device
            count = row[1]     # Second column is count
            
            # Now get all events for this device
            self.cursor.execute("""
                SELECT id, user_id, data, created_at FROM mouse_events
                WHERE data->>'device' = %s
            """, (device,))
            events = self.cursor.fetchall()
            
            aggregated_results.append({
                'device': device,
                'count': count,
                'events': events
            })
        
        return aggregated_results

    # 2. Update operations
    def update_event(self, event_id, update_data):
        """Update a single event."""
        query = """
        UPDATE mouse_events
        SET data = data || %s
        WHERE data->>'event_id' = %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (json.dumps(update_data), event_id))
            self.conn.commit()
            return cur.rowcount

    def update_events_batch(self, event_ids, update_data):
        """Update multiple events in a single operation."""
        query = """
        UPDATE mouse_events
        SET data = data || %s
        WHERE data->>'event_id' IN %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (json.dumps(update_data), tuple(event_ids)))
            self.conn.commit()
            return cur.rowcount

    def update_events_conditional(self, conditions, update_data):
        """Update events that match certain conditions."""
        conditions_sql = []
        params = []
        
        for key, value in conditions.items():
            conditions_sql.append(f"data->>'{key}' = %s")
            params.append(value)
        
        where_clause = " AND ".join(conditions_sql)
        query = f"""
        UPDATE mouse_events
        SET data = data || %s
        WHERE {where_clause}
        """
        
        params.insert(0, json.dumps(update_data))
        
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            self.conn.commit()
            return cur.rowcount

    # 3. Delete operations
    def delete_event(self, event_id):
        """Delete a single event."""
        query = """
        DELETE FROM mouse_events
        WHERE data->>'event_id' = %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (event_id,))
            self.conn.commit()
            return cur.rowcount

    def delete_events_batch(self, event_ids):
        """Delete multiple events in a single operation."""
        query = """
        DELETE FROM mouse_events
        WHERE data->>'event_id' IN %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, (tuple(event_ids),))
            self.conn.commit()
            return cur.rowcount

    def delete_events_conditional(self, conditions):
        """Delete events that match certain conditions."""
        conditions_sql = []
        params = []
        
        for key, value in conditions.items():
            conditions_sql.append(f"data->>'{key}' = %s")
            params.append(value)
        
        where_clause = " AND ".join(conditions_sql)
        query = f"""
        DELETE FROM mouse_events
        WHERE {where_clause}
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            self.conn.commit()
            return cur.rowcount