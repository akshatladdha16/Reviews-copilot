from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
from database import database
from ai_service import ai_service
from pgvector.asyncpg import register_vector
import logging
logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        self.review_texts = []
    
    async def build_tfidf_index(self):
        """Build TF-IDF index from all reviews"""
        async with database.pool.acquire() as conn:
            reviews = await conn.fetch('SELECT id,location,text FROM reviews WHERE text IS NOT NULL')
        
        self.review_texts = [review['text'] for review in reviews]
        
        if self.review_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.review_texts)
        
        return len(self.review_texts)
    
    async def tfidf_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using TF-IDF and cosine similarity"""
        try:
            # Check if index needs to be built
            if self.tfidf_matrix is None or len(self.review_texts) == 0:
                await self.build_tfidf_index()
            
            if len(self.review_texts) == 0:
                return []
            
            # Transform input query to TF-IDF vector
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top K results or if we want we can resort to k>0.5 similarity threshold
            indices = np.argsort(similarities)[-k:][::-1] 
            
            # Get full review data for results
            async with database.pool.acquire() as conn:
                results = []
                for idx in indices:
                    if similarities[idx] > 0:  # Only include relevant results
                        review_id = (await conn.fetchrow(
                            'SELECT id FROM reviews WHERE text = $1', 
                            self.review_texts[idx]
                        ))['id']
                        
                        review = await conn.fetchrow('''
                            SELECT 
                                id, 
                                location,
                                rating,
                                text,
                                date,
                                sentiment,
                                topic,
                                embedding::float[] as embedding
                            FROM reviews 
                            WHERE id = $1
                        ''', review_id)
                        
                        if review:
                            result_dict = dict(review)
                            # Convert embedding to list if it's a string
                            if isinstance(result_dict.get('embedding'), str):
                                try:
                                    # Remove brackets and split by comma
                                    embedding_str = result_dict['embedding'].strip('[]')
                                    result_dict['embedding'] = [
                                        float(x.strip()) 
                                        for x in embedding_str.split(',') 
                                        if x.strip()
                                    ]
                                except (ValueError, AttributeError):
                                    result_dict['embedding'] = None
                            
                            results.append(result_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in TFIDF search: {str(e)}")
            return []
        
        
    async def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector embeddings"""
        # Generate query embedding
        try:
            query_embedding = ai_service.get_embedding(query) #input query converted to vector
        
            async with database.pool.acquire() as conn:
                await register_vector(conn) 
                # await conn.execute('SELECT $1::vector', query_embedding)
            
                results = await conn.fetch('''
                SELECT 
                    id, 
                    location,
                    rating,
                    text,
                    date,
                    sentiment,
                    topic,
                    1 - (embedding <=> $1::vector(384)) as similarity
                FROM reviews 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector(384)
                LIMIT $2
            ''', query_embedding, k)
            
            # Filter and convert to dict
                return[
                    {
                        **dict(row),
                        'similarity': float(row['similarity']) if row['similarity'] is not None else None

                    }for row in results
                ]
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise
    async def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Combine TF-IDF and vector search results"""
        tfidf_results = await self.tfidf_search(query, k)
        vector_results = await self.vector_search(query, k)
        
        # Combine and deduplicate results
        all_results = {}
        for result in tfidf_results + vector_results:
            all_results[result['id']] = result
        
        return list(all_results.values())[:k]

# Global search service instance
search_service = SearchService()