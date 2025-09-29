import asyncpg # non blocking postgres library so that we can use it in connection pool
from pgvector.asyncpg import register_vector
import os
from typing import List, Optional
import json
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

class Database:
    # initialize the database connection pool so that not every time a new connection is created.
    def __init__(self): #consturctor sync by nature 
        self.pool = None 
# so we need to create a async init method to create connection pool 
    async def init(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(os.getenv('POSTGRES_URI'))
        async with self.pool.acquire() as conn:
            await register_vector(conn) # now we can use vector type and store reviews embeddinnsg for semantic search
            await self._create_tables(conn)

    async def _create_tables(self, conn):
        """Create necessary tables"""
         # Create pgvector extension first
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
        
        # Wait a moment to ensure extension is fully loaded
        await conn.execute('SELECT 1')
        await register_vector(conn) #register vector type with asyncpg
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id SERIAL PRIMARY KEY,
                location VARCHAR(100) NOT NULL,
                rating INTEGER CHECK (rating >= 1 AND rating <= 5) NOT NULL,
                text TEXT NOT NULL,
                date DATE NOT NULL,
                sentiment VARCHAR(20),
                topic VARCHAR(50),
                embedding VECTOR(384), 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_reviews_location ON reviews(location);
            CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment);
            CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
            CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(date);
            CREATE INDEX IF NOT EXISTS idx_reviews_embedding ON reviews USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        ''') #using cosine distance for similarity search
        # creates a B-Tree indexing , used indirectly so that it makes a key value pair for faster searching, now the SQL query will not search one by one but will use the index to directly go to the location where the data is stored.

    async def insert_review(self, review_data: dict) -> int:
        """Insert a single review and return its ID"""
        async with self.pool.acquire() as conn: # acquire a connection from the pool
            result = await conn.fetchrow('''
                INSERT INTO reviews (location, rating, text, date, sentiment, topic, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            ''', 
            review_data['location'], review_data['rating'], review_data['text'],
            review_data['date'], review_data.get('sentiment'), review_data.get('topic'),
            review_data.get('embedding'))
            return result['id']

    async def bulk_insert_reviews(self, reviews: List[dict]) -> List[int]:
        """Insert multiple reviews efficiently"""
        async with self.pool.acquire() as conn:
            ids = []
            try:
                # Start a transaction
                async with conn.transaction():
                    for review in reviews:
                        try:
                            # Convert date string to proper format if needed
                            if isinstance(review['date'], str):
                                review['date'] = datetime.strptime(review['date'], '%Y-%m-%d').date()
                            
                            result = await conn.fetchrow('''
                                INSERT INTO reviews (location, rating, text, date)
                                VALUES ($1, $2, $3, $4)
                                RETURNING id
                            ''', review['location'], review['rating'], review['text'], review['date'])
                            ids.append(result['id'])
                        except Exception as e:
                            logger.error(f"Error inserting review: {review}, Error: {str(e)}")
                            raise
                    
                    return ids
            except Exception as e:
                logger.error(f"Bulk insert failed: {str(e)}", exc_info=True)
                raise
    #adding unique locations to avoid manual entry in 
    async def get_unique_locations(self) -> List[str]:
        """Get all unique locations from reviews"""
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT location 
                    FROM reviews 
                    WHERE location IS NOT NULL 
                    ORDER BY location
                    """
                )
                return [row['location'] for row in rows]
            except Exception as e:
                logger.error(f"Error fetching unique locations: {str(e)}")
                raise
    async def get_review(self, review_id: int) -> Optional[dict]:
        """Get a single review by ID"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow('''
                SELECT * FROM reviews WHERE id = $1
            ''', review_id)
            return dict(result) if result else None

    async def get_reviews(self, location: str = None, sentiment: str = None, 
                         rating: int = None, page: int = 1, page_size: int = 20) -> tuple:
        """Get reviews with filtering and pagination, based on location, ratings and sentiments as requested"""
        async with self.pool.acquire() as conn:
            # Build WHERE clause dynamically
            conditions = []
            params = []
            param_count = 0 #no. of paramters added for filters and then we appply that in the rows. 
            
            if location:
                param_count += 1
                conditions.append(f"location = ${param_count}")
                params.append(location)
            
            if sentiment:
                param_count += 1
                conditions.append(f"sentiment = ${param_count}")
                params.append(sentiment)
            
            if rating:
                param_count += 1
                conditions.append(f"rating = ${param_count}")
                params.append(rating)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1" # to keep query valid
            
            # Get total count
            count_result = await conn.fetchrow(f'''
                SELECT COUNT(*) as total FROM reviews WHERE {where_clause}
            ''', *params)
            total = count_result['total']
            
            # Get paginated results
            param_count += 1
            params.append(page_size)
            param_count += 1
            params.append((page - 1) * page_size)
            
            results = await conn.fetch(f'''
                SELECT id, 
                    location, 
                    rating, 
                    text, 
                    date::text as date,
                    sentiment, 
                    topic,
                    created_at 
                    FROM reviews 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_count - 1} OFFSET ${param_count}
            ''', *params) # limits the page size and offset to get data only for the current page and skips rows for earlier pages. eg page 3 , page size 20. so we have to skip 40 so offset becomes 40
            
            return [dict(row) for row in results], total

    async def update_review_sentiment(self, review_id: int, sentiment: str, topic: str):
        """Update review with AI-generated sentiment and topic"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE reviews 
                SET sentiment = $1, topic = $2, updated_at = CURRENT_TIMESTAMP
                WHERE id = $3
            ''', sentiment, topic, review_id)

    async def update_review_embedding(self, review_id: int, embedding: List[float]):
        """Update review with text embedding"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE reviews 
                SET embedding = $1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
            ''', embedding, review_id)

    async def get_analytics(self) -> dict:
        """Get analytics data for dashboard"""
        try:
            async with self.pool.acquire() as conn:
                #avg ratings
                avg_rating = await conn.fetchval('''
                    SELECT ROUND(AVG(rating)::numeric, 2)
                    FROM reviews
                    WHERE rating IS NOT NULL
                ''')
                # Sentiment counts
                sentiment_results = await conn.fetch('''
                    SELECT sentiment, COUNT(*) as count 
                    FROM reviews 
                    WHERE sentiment IS NOT NULL 
                    GROUP BY sentiment
                ''')
                sentiment_counts = {row['sentiment']: row['count'] for row in sentiment_results}
                
                # Topic counts
                topic_results = await conn.fetch('''
                    SELECT topic, COUNT(*) as count 
                    FROM reviews 
                    WHERE topic IS NOT NULL 
                    GROUP BY topic
                ''')
                topic_counts = {row['topic']: row['count'] for row in topic_results}
                
                # Location counts
                location_results = await conn.fetch('''
                    SELECT location, COUNT(*) as count 
                    FROM reviews 
                    GROUP BY location
                ''')
                location_counts = {row['location']: row['count'] for row in location_results}
                
                return {
                    'average_rating': float(avg_rating) if avg_rating is not None else None,
                    'sentiment_counts': dict(sentiment_counts),
                    'topic_counts': dict(topic_counts),
                    'location_counts': dict(location_counts)
                }
        except Exception as e:
            logger.error(f"Error fetching analytics: {str(e)}")
            raise

# Database instance
database = Database()