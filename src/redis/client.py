#!/usr/bin/env python3
"""
Module to interact with Redis for the mouse tracking benchmark.
"""
import redis
import json
import time
from datetime import datetime

class RedisClient:
    """Client for interacting with Redis."""
    
    def __init__(self, host='localhost', port=6379, db=0, prefix='mouse:event:'):
        """Initialize the connection to Redis."""
        self.client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
        self.user_index_key = 'mouse:user:index'
        self.page_index_key = 'mouse:page:index'
        self.counter_key = 'mouse:events:counter'
        
    def _generate_key(self, event_id):
        """Generate a Redis key for an event."""
        return f"{self.prefix}{event_id}"
        
    def insert_event(self, event):
        """Insert a mouse event in Redis."""
        # Get and increment the counter for ID generation
        event_id = self.client.incr(self.counter_key)
        key = self._generate_key(event_id)
        
        # Store event data
        self.client.set(key, json.dumps(event))
        
        # Add to user index (set of event IDs for each user)
        user_id = event.get('user_id')
        user_index_key = f"{self.user_index_key}:{user_id}"
        self.client.sadd(user_index_key, event_id)
        
        # Add to page index (set of event IDs for each page)
        page = event.get('page')
        if page:
            page_index_key = f"{self.page_index_key}:{page}"
            self.client.sadd(page_index_key, event_id)
            
        return event_id
        
    def insert_events(self, events):
        """Insert multiple mouse events in Redis."""
        # Use a pipeline for better performance
        pipe = self.client.pipeline()
        event_ids = []
        
        for event in events:
            # Get and increment the counter for ID generation
            event_id = self.client.incr(self.counter_key)
            event_ids.append(event_id)
            key = self._generate_key(event_id)
            
            # Store event data
            pipe.set(key, json.dumps(event))
            
            # Add to user index
            user_id = event.get('user_id')
            user_index_key = f"{self.user_index_key}:{user_id}"
            pipe.sadd(user_index_key, event_id)
            
            # Add to page index
            page = event.get('page')
            if page:
                page_index_key = f"{self.page_index_key}:{page}"
                pipe.sadd(page_index_key, event_id)
        
        # Execute all commands in the pipeline
        pipe.execute()
        return event_ids
        
    def find_events_by_user(self, user_id):
        """Get all events from a user."""
        user_index_key = f"{self.user_index_key}:{user_id}"
        event_ids = self.client.smembers(user_index_key)
        
        if not event_ids:
            return []
            
        events = []
        for event_id in event_ids:
            key = self._generate_key(event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id)
            event_data = self.client.get(key)
            if event_data:
                event = json.loads(event_data)
                event['id'] = event_id
                events.append(event)
                
        return events
        
    def find_events_by_page(self, page):
        """Get all events from a specific page."""
        page_index_key = f"{self.page_index_key}:{page}"
        event_ids = self.client.smembers(page_index_key)
        
        if not event_ids:
            return []
            
        events = []
        for event_id in event_ids:
            key = self._generate_key(event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id)
            event_data = self.client.get(key)
            if event_data:
                event = json.loads(event_data)
                event['id'] = event_id
                events.append(event)
                
        return events
        
    def count_events(self):
        """Count the total number of events."""
        return int(self.client.get(self.counter_key) or 0)
        
    def clear_data(self):
        """Clear all mouse tracking related data."""
        # Get all keys matching the patterns
        keys = []
        keys.extend(self.client.keys(f"{self.prefix}*"))
        keys.extend(self.client.keys(f"{self.user_index_key}:*"))
        keys.extend(self.client.keys(f"{self.page_index_key}:*"))
        keys.append(self.counter_key)
        
        if keys:
            # Delete all keys in a single operation
            self.client.delete(*keys)
            
    def close(self):
        """Close the connection to Redis."""
        self.client.close()
        
    def __enter__(self):
        """Support for context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up."""
        self.close()