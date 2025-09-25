from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
from database import database
from ai_service import ai_service

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
            reviews = await conn.fetch('SELECT id, text FROM reviews WHERE text IS NOT NULL')
        
        self.review_texts = [review['text'] for review in reviews]
        
        if self.review_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.review_texts)
        
        return len(self.review_texts)
    
    async def tfidf_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using TF-IDF and cosine similarity"""
        # Check if index needs to be built
        if self.tfidf_matrix is None or len(self.review_texts) == 0:
            await self.build_tfidf_index()
        
        if len(self.review_texts) == 0:
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vec = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top K results
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
                        
                        review = await conn.fetchrow(
                            'SELECT * FROM reviews WHERE id = $1', 
                            review_id
                        )
                        if review:
                            results.append(dict(review))
            
            return results
        except Exception as e:
            logger.error(f"Error in TFIDF search: {str(e)}")
            return []
        
        
    async def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector embeddings"""
        # Generate query embedding
        query_embedding = ai_service.get_embedding(query)
        
        async with database.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT *, 
                       (1 - (embedding <=> $1)) as similarity
                FROM reviews 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1
                LIMIT $2
            ''', query_embedding, k)
        
        return [dict(row) for row in results if row['similarity'] > 0.5]  # Threshold for relevance
    
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