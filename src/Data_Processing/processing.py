import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text_with_stopwords(news):
    """Clean text by removing unwanted characters and stopwords."""
    text = news.lower()
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)

def perform_word_frequency_analysis(news_df):
    """Perform word frequency analysis on cleaned text."""
    all_words = ' '.join(news_df['cleaned_text']).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(10)

def vectorize_text_data(news_df):
    """Convert cleaned text into numerical representations using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(news_df['cleaned_text'])
    return X.shape

def process_dates(news_df):
    """Convert 'date' column to datetime and extract useful time features."""
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df['month'] = news_df['date'].dt.month
    news_df['year'] = news_df['date'].dt.year
    return news_df[['date', 'month', 'year']]


# Convert sentiment to binary labels efficiently
news_df['sentiment_label'] = (news_df['sentiment'] == 'POSITIVE').astype(int)

# Clean text by removing unwanted characters and stopwords
news_df['cleaned_text'] = news_df['news'].apply(clean_text_with_stopwords)

# Perform word frequency analysis after cleaning the text
most_common_words = perform_word_frequency_analysis(news_df)
print("Most common words after processing: ", most_common_words)

# Convert cleaned text into numerical features using TF-IDF
X_shape = vectorize_text_data(news_df)
print("Shape of the transformed matrix: ", X_shape)

# Process dates to extract month and year
time_features = process_dates(news_df)
print("After processing dates: \n", time_features.head())
