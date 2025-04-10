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
    
    # 1. Requêtes avancées
    def find_events_by_time_range(self, start_date, end_date):
        """Trouve les événements dans une plage de temps spécifique."""
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        all_keys = self.redis.keys('mouse_event:*')
        result = []
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                if 'timestamp' in event:
                    if start_str <= event['timestamp'] <= end_str:
                        result.append(event)
        
        return result

    def find_events_in_screen_zone(self, x_min, x_max, y_min, y_max):
        """Trouve les événements dans une zone d'écran spécifique."""
        all_keys = self.redis.keys('mouse_event:*')
        result = []
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                x_pos = float(event.get('x_pos', 0))
                y_pos = float(event.get('y_pos', 0))
                
                if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
                    result.append(event)
        
        return result

    def aggregate_events_by_user(self):
        """Agrège les événements par utilisateur."""
        all_keys = self.redis.keys('mouse_event:*')
        user_events = {}
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                user_id = event.get('user_id')
                
                if user_id:
                    if user_id not in user_events:
                        user_events[user_id] = []
                    user_events[user_id].append(event)
        
        result = []
        for user_id, events in user_events.items():
            result.append({
                'user_id': user_id,
                'count': len(events),
                'events': events
            })
        
        return result

    def aggregate_events_by_page(self):
        """Agrège les événements par page."""
        all_keys = self.redis.keys('mouse_event:*')
        page_events = {}
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                page = event.get('page')
                
                if page:
                    if page not in page_events:
                        page_events[page] = []
                    page_events[page].append(event)
        
        result = []
        for page, events in page_events.items():
            result.append({
                'page': page,
                'count': len(events),
                'events': events
            })
        
        return result

    def aggregate_events_by_device(self):
        """Agrège les événements par type de dispositif."""
        all_keys = self.redis.keys('mouse_event:*')
        device_events = {}
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                device = event.get('device')
                
                if device:
                    if device not in device_events:
                        device_events[device] = []
                    device_events[device].append(event)
        
        result = []
        for device, events in device_events.items():
            result.append({
                'device': device,
                'count': len(events),
                'events': events
            })
        
        return result

    # 2. Opérations de mise à jour
    def update_event(self, event_id, update_data):
        """Met à jour un seul événement."""
        key = f'mouse_event:{event_id}'
        event_data = self.redis.get(key)
        
        if event_data:
            event = json.loads(event_data)
            event.update(update_data)
            self.redis.set(key, json.dumps(event))
            return 1
        
        return 0

    def update_events_batch(self, event_ids, update_data):
        """Met à jour plusieurs événements en une seule opération."""
        count = 0
        
        for event_id in event_ids:
            if self.update_event(event_id, update_data):
                count += 1
        
        return count

    def update_events_conditional(self, conditions, update_data):
        """Met à jour les événements qui correspondent à certaines conditions."""
        all_keys = self.redis.keys('mouse_event:*')
        count = 0
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                
                # Vérifier si l'événement correspond aux conditions
                match = True
                for cond_key, cond_value in conditions.items():
                    if event.get(cond_key) != cond_value:
                        match = False
                        break
                
                if match:
                    event.update(update_data)
                    self.redis.set(key, json.dumps(event))
                    count += 1
        
        return count

    # 3. Opérations de suppression
    def delete_event(self, event_id):
        """Supprime un seul événement."""
        key = f'mouse_event:{event_id}'
        if self.redis.exists(key):
            self.redis.delete(key)
            return 1
        return 0

    def delete_events_batch(self, event_ids):
        """Supprime plusieurs événements en une seule opération."""
        keys = [f'mouse_event:{event_id}' for event_id in event_ids]
        count = 0
        
        for key in keys:
            if self.redis.exists(key):
                self.redis.delete(key)
                count += 1
        
        return count

    def delete_events_conditional(self, conditions):
        """Supprime les événements qui correspondent à certaines conditions."""
        all_keys = self.redis.keys('mouse_event:*')
        count = 0
        
        for key in all_keys:
            event_data = self.redis.get(key)
            if event_data:
                event = json.loads(event_data)
                
                # Vérifier si l'événement correspond aux conditions
                match = True
                for cond_key, cond_value in conditions.items():
                    if event.get(cond_key) != cond_value:
                        match = False
                        break
                
                if match:
                    self.redis.delete(key)
                    count += 1
        
        return count