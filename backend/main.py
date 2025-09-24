from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Optional
import json
from datetime import datetime

# from .database import database
from database import database
from models import *
from search_rag import search_service
import ai_service
from ai_service import *
from auth import require_admin, require_analytics, get_user
from cache import cache

app = FastAPI(
    title="Reviews Copilot API",
    description="AI-powered customer reviews management system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Startup event
@app.on_event("startup")
async def startup_event():
    # Initialize cache
    # cache.init_redis()
    
    # Initialize database
    await database.init() # custom async init function 
    
    # Build search index
    await search_service.build_tfidf_index()
    print("Application startup completed")

# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Ingest endpoints
@app.post("/ingest", response_model=dict, dependencies=[Depends(require_admin)])
async def ingest_reviews(file: UploadFile = File(...)):
    """Ingest reviews from JSON file"""
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are supported")
        
        # Read and parse JSON file
        content = await file.read()
        reviews_data = json.loads(content)
        
          # Log the received data
        logger.info(f"Received {len(reviews_data)} reviews to ingest")
        

        # Validate reviews
        if not isinstance(reviews_data, list):
            raise HTTPException(status_code=400, detail="JSON should contain an array of reviews")
        
        # Insert reviews
        review_ids = await database.bulk_insert_reviews(reviews_data)
        
        # Process reviews with AI (async, don't wait)
        await process_reviews_with_ai(review_ids)
        
        return {
            "message": f"Successfully ingested {len(review_ids)} reviews",
            "review_ids": review_ids
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException as he:
        logger.error(f"HTTP error during ingestion: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ingesting reviews: {str(e)}")

async def process_reviews_with_ai(review_ids: List[int]):
    """Process reviews with AI in background"""
    for review_id in review_ids:
        try:
            review = await database.get_review(review_id)
            if review:
                # Analyze sentiment and topic
                sentiment = ai_service.analyze_sentiment(review['text'])
                topic = ai_service.detect_topic(review['text'])
                
                # Generate embedding
                embedding = ai_service.get_embedding(review['text'])
                
                # Update review with AI data
                await database.update_review_sentiment(review_id, sentiment, topic)
                await database.update_review_embedding(review_id, embedding)
                
        except Exception as e:
            print(f"Error processing review {review_id}: {e}")

# Review management endpoints
@app.get("/reviews", response_model=PaginatedResponse)
async def get_reviews(
    location: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    rating: Optional[int] = Query(None, ge=1, le=5),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict = Depends(get_user)
):
    """Get paginated reviews with filtering"""
    reviews, total = await database.get_reviews(
        location=location,
        sentiment=sentiment,
        rating=rating,
        page=page,
        page_size=page_size
    )
    
    total_pages = (total + page_size - 1) // page_size
    
    return PaginatedResponse(
        data=reviews,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )

@app.get("/reviews/{review_id}", response_model=Review)
async def get_review(review_id: int, user: dict = Depends(get_user)):
    """Get a specific review by ID"""
    review = await database.get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review

@app.post("/reviews/{review_id}/suggest-reply", response_model=ReplySuggestion)
async def suggest_reply(review_id: int, user: dict = Depends(require_admin)):
    """Generate AI-suggested reply for a review"""
    review = await database.get_review(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    suggestion = await ai_service.suggest_reply(
        review['text'],
        review['rating'],
        review['location']
    )
    
    return ReplySuggestion(**suggestion)

# Analytics endpoint
@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(user: dict = Depends(require_analytics)):
    """Get analytics data"""
    analytics_data = await database.get_analytics()
    return AnalyticsResponse(**analytics_data)

# Search endpoints
@app.get("/search", response_model=SearchResponse)
async def search_reviews(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
    search_type: str = Query("hybrid", regex="^(tfidf|vector|hybrid)$")
):
    """Search reviews using different methods"""
    if search_type == "tfidf":
        results = await search_service.tfidf_search(q, k)
    elif search_type == "vector":
        results = await search_service.vector_search(q, k)
    else:  # hybrid
        results = await search_service.hybrid_search(q, k)
    
    return SearchResponse(
        reviews=results,
        search_type=search_type,
        query=q
    )

# Cache management endpoints (admin only)
# @app.delete("/cache/clear", dependencies=[Depends(require_admin)])
# async def clear_cache():
#     """Clear all cache (admin only)"""
#     # Note: This is a simplified implementation
#     # In production, you might want more specific cache clearing
#     cache.local_cache.clear()
#     return {"message": "Cache cleared successfully"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)