import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from dotenv import load_dotenv
load_dotenv() #streamlit takes secrets not env , need to define them in streamlit secrets. 

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY="admin123"
# Page configuration
st.set_page_config(
    page_title="Reviews Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .review-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .negative-review {
        border-left-color: #ff4b4b;
    }
    .positive-review {
        border-left-color: #00d154;
    }
</style>
""", unsafe_allow_html=True)

class ReviewsClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def _make_request(self, endpoint, method="GET", data=None):
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=data)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    def health_check(self):
        return self._make_request("/health")
    
    def get_locations(self):
        """Fetch all unique locations"""
        return self._make_request("/locations") 
    
    def get_reviews(self, location=None, sentiment=None, rating=None, page=1, page_size=20):
        params = {"page": page, "page_size": page_size}
        if location: params["location"] = location
        if sentiment: params["sentiment"] = sentiment
        if rating: params["rating"] = rating
        return self._make_request("/reviews", data=params)
    
    def get_review(self, review_id):
        return self._make_request(f"/reviews/{review_id}")
    
    def suggest_reply(self, review_id):
        return self._make_request(f"/reviews/{review_id}/suggest-reply", method="POST")
    
    def get_analytics(self):
        return self._make_request("/analytics")
    
    def search_reviews(self, query, k=5, search_type="hybrid"):
        return self._make_request("/search", data={"q": query, "k": k, "search_type": search_type})
    
    def ingest_reviews(self, file_content):
        files = {"file": ("reviews.json", file_content, "application/json")}
        try:
            response = requests.post(
                f"{self.base_url}/ingest",
                headers=self.headers,
                files=files
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Upload Error: {e}")
            return None

# Initialize client
client = ReviewsClient(BACKEND_URL, API_KEY)

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Review Inbox", "Analytics", "Search", "Admin"])
    
    # Main content
    st.markdown('<div class="main-header">📊 Reviews Copilot</div>', unsafe_allow_html=True)
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Review Inbox":
        show_review_inbox()
    elif page == "Analytics":
        show_analytics()
    elif page == "Search":
        show_search()
    elif page == "Admin":
        show_admin()

def show_dashboard():
    st.header("Dashboard Overview")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    analytics = client.get_analytics()
    if analytics:
        with col1:
            total_reviews = sum(analytics.get('location_counts', {}).values())
            st.metric("Total Reviews", total_reviews)
        
        with col2:
            positive_reviews = analytics.get('sentiment_counts', {}).get('positive', 0)
            st.metric("Positive Reviews", positive_reviews)
        
        with col3:
            negative_reviews = analytics.get('sentiment_counts', {}).get('negative', 0)
            st.metric("Negative Reviews", negative_reviews)
        
        with col4:
            avg_rating = calculate_average_rating(analytics.get('rating_distribution', {}))
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
    
    # Recent reviews
    st.subheader("Recent Reviews")
    reviews = client.get_reviews(page_size=10)
    if reviews and reviews.get('data'):
        for review in reviews['data'][:5]:
            display_review_card(review)
    else:
        st.info("No reviews found. Upload some reviews to get started.")

def show_review_inbox():
    st.header("Review Inbox")
    # Fetch available locations
    locations_response = client.get_locations()
    available_locations = ["All"] + (locations_response if locations_response else [])
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        location_filter = st.selectbox(
            "Filter by Location",
            options=available_locations,
            key="location_filter"
        )
    
    with col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
    
    with col3:
        rating_filter = st.selectbox("Rating", ["All", "1", "2", "3", "4", "5"])
    
    with col4:
        items_per_page = st.selectbox("Items per page", [10, 20, 50])
    
    # Pagination
    page = st.number_input("Page", min_value=1, value=1)
    
    # Load reviews
    params = {}
    if location_filter != "All": params["location"] = location_filter
    if sentiment_filter != "All": params["sentiment"] = sentiment_filter
    if rating_filter != "All": params["rating"] = int(rating_filter)
    
    reviews_data = client.get_reviews(page=page, page_size=items_per_page, **params)
    
    if reviews_data and reviews_data.get('data'):
        st.write(f"Showing {len(reviews_data['data'])} of {reviews_data['total']} reviews")
        
        for review in reviews_data['data']:
            with st.expander(f"Review #{review['id']} - {review['location']} - {review['rating']}⭐"):
                display_review_detail(review)
    else:
        st.info("No reviews match your filters.")

def show_analytics():
    st.header("Analytics Dashboard")
    
    analytics = client.get_analytics()
    if not analytics:
        st.error("Failed to load analytics data")
        return
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Topics", "Locations", "Ratings"])
    
    with tab1:
        st.subheader("Sentiment Analysis")
        sentiment_data = analytics.get('sentiment_counts', {})
        if sentiment_data:
            fig = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="Review Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available")
    
    with tab2:
        st.subheader("Topic Analysis")
        topic_data = analytics.get('topic_counts', {})
        if topic_data:
            fig = px.bar(
                x=list(topic_data.keys()),
                y=list(topic_data.values()),
                title="Review Topics Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available")
    
    with tab3:
        st.subheader("Location Analysis")
        location_data = analytics.get('location_counts', {})
        if location_data:
            fig = px.bar(
                x=list(location_data.keys()),
                y=list(location_data.values()),
                title="Reviews by Location"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available")
    
    with tab4:
        st.subheader("Rating Distribution")
        rating_data = analytics.get('rating_distribution', {})
        if rating_data:
            fig = px.bar(
                x=list(rating_data.keys()),
                y=list(rating_data.values()),
                title="Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rating data available")

def show_search():
    st.header("Advanced Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search reviews", placeholder="Enter keywords to search...")
    
    with col2:
        search_type = st.selectbox("Search type", ["hybrid", "tfidf", "vector"])
    
    if search_query:
        results = client.search_reviews(search_query, search_type=search_type)
        
        if results and results.get('reviews'):
            st.write(f"Found {len(results['reviews'])} similar reviews")
            
            for review in results['reviews']:
                display_review_card(review)
        else:
            st.info("No similar reviews found.")

def show_admin():
    st.header("Admin Panel")
    
    tab1, tab2, tab3 = st.tabs(["Upload Reviews", "System Health", "Cache Management"])
    
    with tab1:
        st.subheader("Upload Reviews")
        
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        
        if uploaded_file is not None:
            if st.button("Upload Reviews"):
                with st.spinner("Uploading reviews..."):
                    result = client.ingest_reviews(uploaded_file.getvalue())
                    
                if result:
                    st.success(f"Successfully uploaded {len(result.get('review_ids', []))} reviews")
                    st.json(result)
    
    with tab2:
        st.subheader("System Health")
        
        if st.button("Check Health"):
            health = client.health_check()
            if health:
                st.success("System is healthy")
                st.json(health)
            else:
                st.error("System health check failed")
    
    with tab3:
        st.subheader("Cache Management")
        
        if st.button("Clear Cache"):
            # This would call a cache clearing endpoint
            st.info("Cache cleared successfully")

def display_review_card(review):
    """Display a review in a card format"""
    # Generate unique IDs for each button
    suggest_btn_key = f"suggest_btn_{review['id']}_{review.get('location', '')}"
    copy_btn_key = f"copy_btn_{review['id']}_{review.get('location', '')}"
    
    sentiment_class = "positive-review" if review.get('sentiment') == 'positive' else "negative-review" if review.get('sentiment') == 'negative' else ""
    
    st.markdown(f'<div class="review-card {sentiment_class}">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.write(f"**Location:** {review.get('location', 'N/A')}")
        st.write(f"**Rating:** {review.get('rating', 'N/A')}⭐")
    
    with col2:
        st.write(f"**Sentiment:** {review.get('sentiment', 'N/A')}")
        st.write(f"**Topic:** {review.get('topic', 'N/A')}")
    
    with col3:
        st.write(f"**Review:** {review.get('text', 'N/A')}")
        
        if st.button("Suggest Reply", key=suggest_btn_key):
            with st.spinner("Generating reply..."):
                suggestion = client.suggest_reply(review['id'])
            
            if suggestion:
                st.text_area("Suggested Reply", suggestion.get('reply', ''), height=100, key=f"reply_{review['id']}")
                
                if st.button("Copy to Clipboard", key=copy_btn_key):
                    st.code(suggestion.get('reply', ''))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
def display_review_detail(review):
    """Display detailed review view"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**ID:** {review['id']}")
        st.write(f"**Location:** {review['location']}")
        st.write(f"**Rating:** {review['rating']}⭐")
        st.write(f"**Date:** {review['date']}")
    
    with col2:
        st.write(f"**Sentiment:** {review.get('sentiment', 'Not analyzed')}")
        st.write(f"**Topic:** {review.get('topic', 'Not analyzed')}")
        st.write(f"**Created:** {review.get('created_at', 'Unknown')}")
    
    st.write(f"**Review Text:**")
    st.write(review['text'])
    
    # AI Reply Suggestion
    st.subheader("AI Reply Suggestion")
    
    if st.button("Generate Reply", key=f"detail_btn_{review['id']}"):
        with st.spinner("AI is generating a reply..."):
            suggestion = client.suggest_reply(review['id'])
        
        if suggestion:
            st.text_area("Suggested Reply", suggestion.get('reply', ''), height=150)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Copy Reply", key=f"copy_detail_{review['id']}"):
                    st.code(suggestion.get('reply', ''))
            with col2:
                if st.button("Edit Reply", key=f"edit_{review['id']}"):
                    st.session_state[f"editing_{review['id']}"] = True
            
            # Show tags and reasoning
            with st.expander("AI Analysis Details"):
                st.json(suggestion.get('tags', {}))
                st.write(f"**Reasoning:** {suggestion.get('reasoning_log', '')}")

def calculate_average_rating(rating_distribution):
    """Calculate average rating from distribution"""
    if not rating_distribution:
        return 0
    
    total = 0
    count = 0
    for rating, freq in rating_distribution.items():
        total += int(rating) * freq
        count += freq
    
    return total / count if count > 0 else 0

if __name__ == "__main__":
    main()