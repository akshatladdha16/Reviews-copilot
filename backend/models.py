from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime,date
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Topic(str, Enum):
    SERVICE = "service"
    FOOD = "food"
    CLEANLINESS = "cleanliness"
    PRICE = "price"
    DELIVERY = "delivery"
    OTHER = "other"
    # can add more topics as needed

class ReviewBase(BaseModel):
    location: str = Field(..., description="Business location") ## need to be filled, no default value given from our side for authentic reviews 
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    text: str = Field(..., description="Review text")
    date: str = Field(..., description="Review date YYYY-MM-DD")

class ReviewCreate(ReviewBase):
    pass ## inherits from ReviewBase class without changes

class Review(ReviewBase):
    id: int
    sentiment: Optional[Sentiment] = None
    topic: Optional[Topic] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    @field_validator('date', mode='before')
    @classmethod
    def validate_date(cls, v):
        if isinstance(v, date):
            return v.isoformat()
        if isinstance(v, datetime):
            return v.date().isoformat()
        return v

    class Config:
        from_attributes = True # now pydantic allows the db model to accept the data in obbject format too not just dict. 

class ReplySuggestion(BaseModel):
    reply: str
    tags: Dict[str, str]
    reasoning_log: str

class AnalyticsResponse(BaseModel):
    average_rating: Optional[float] = Field(
        None, 
        description="Average rating of all reviews",
        ge=1.0,
        le=5.0
    )
    sentiment_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of reviews by sentiment"
    )
    topic_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of reviews by topic"
    )
    location_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of reviews by location"
    )

    @field_validator('average_rating')
    @classmethod
    def validate_rating(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(float(v), 2)
        return None

class SearchResponse(BaseModel):
    reviews: List[Review]
    search_type: str
    query: str

class PaginatedResponse(BaseModel):
    data: List[Review]
    total: int
    page: int
    page_size: int
    total_pages: int