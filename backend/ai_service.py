import os
from dotenv import load_dotenv
load_dotenv()
import groq
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import re
# import cache
import models
# from cache import cached
from models import Sentiment, Topic

class AIService:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-defined topics for classification
        self.topics_keywords = {
            Topic.SERVICE: ['service', 'staff', 'employee', 'worker', 'helpful', 'friendly', 'rude'],
            Topic.FOOD: ['food', 'taste', 'delicious', 'meal', 'dish', 'flavor', 'menu'],
            Topic.CLEANLINESS: ['clean', 'dirty', 'hygiene', 'tidy', 'messy', 'sanitary'],
            Topic.PRICE: ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value'],
            Topic.DELIVERY: ['delivery', 'deliver', 'shipping', 'arrive', 'late', 'early', 'driver']
        }
    
    # @cached(expire=86400, key_prefix="embedding")  # Cache for 24 hours
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching"""
        return self.embedding_model.encode(text).tolist()
    
    # @cached(expire=3600, key_prefix="sentiment")
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment using Groq"""
        try:
            prompt = f"""
            Analyze the sentiment of this customer review. Return only one word: positive, negative, or neutral.
            
            Review: "{text}"
            
            Sentiment:
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Fast and efficient model
                max_tokens=10,
                temperature=0.1
            )
            
            sentiment = response.choices[0].message.content.strip().lower()
            return sentiment if sentiment in ['positive', 'negative', 'neutral'] else 'neutral'
            
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            # Fallback: simple rule-based sentiment
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'awesome', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
    
    def detect_topic(self, text: str) -> str:
        """Detect topic based on keywords"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.topics_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        return Topic.OTHER
    
    def redact_sensitive_info(self, text: str) -> str:
        """Redact emails and phone numbers from text"""
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Redact phone numbers
        text = re.sub(r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b', '[PHONE]', text)
        
        return text
    
    # @cached(expire=1800, key_prefix="reply")  # Cache for 30 minutes
    async def suggest_reply(self, review_text: str, rating: int, location: str) -> Dict[str, Any]:
        """Generate a suggested reply using Groq with safeguards"""
        try:
            # Redact sensitive information
            safe_text = self.redact_sensitive_info(review_text)
            
            # Analyze sentiment and topic
            sentiment = self.analyze_sentiment(safe_text)
            topic = self.detect_topic(safe_text)
            
            # Create context-aware prompt
            prompt = f"""
            You are a customer service representative for a multi-location business.
            Generate a concise, empathetic response to this customer review.
            
            Review Details:
            - Location: {location}
            - Rating: {rating}/5
            - Sentiment: {sentiment}
            - Topic: {topic}
            - Review Text: "{safe_text}"
            
            Guidelines:
            1. Be empathetic and professional
            2. Address the specific concern mentioned
            3. Keep it concise (2-3 sentences max)
            4. For negative reviews, acknowledge the issue and suggest improvement
            5. For positive reviews, express gratitude and encourage return
            6. Never include toxic or defensive language
            7. Do not make promises you can't keep
            8. Keep it natural and human-sounding
            
            Suggested Reply:
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",  # High-quality model for replies
                max_tokens=150,
                temperature=0.7,
                stop=["\n\n"]
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Basic content safety check
            toxic_words = ['stupid', 'idiot', 'hate you', 'terrible company', 'never again']
            if any(word in reply.lower() for word in toxic_words):
                reply = "Thank you for your feedback. We appreciate you bringing this to our attention and will use it to improve our service."
            
            return {
                "reply": reply,
                "tags": {
                    "sentiment": sentiment,
                    "topic": topic,
                    "rating": rating
                },
                "reasoning_log": f"Generated using Groq AI. Sentiment: {sentiment}, Topic: {topic}"
            }
            
        except Exception as e:
            print(f"Reply generation failed: {e}")
            # Fallback reply
            return {
                "reply": "Thank you for your feedback. We appreciate you taking the time to share your experience with us.",
                "tags": {
                    "sentiment": "neutral",
                    "topic": "other",
                    "rating": rating
                },
                "reasoning_log": f"Fallback reply due to error: {str(e)}"
            }

# Global AI service instance
ai_service = AIService()