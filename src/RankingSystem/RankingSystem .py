import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_news_data(news_data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(news_data)

    # Convert sentiment to numerical values
    sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_score"] = df["sentiment"].map(sentiment_mapping)

    # Combine the news impact with sentiment score to determine investment importance
    df["investment_score"] = df["sentiment_score"] * df["impact_factor"]

    # Use MinMaxScaler to normalize the investment scores
    scaler = MinMaxScaler()
    df["scaled_investment_score"] = scaler.fit_transform(df[["investment_score"]])

    # Display the results in the format you requested
    for index, row in df.iterrows():
        print(f"Headline: {row['headline']}")
        print(f"Positive Sentiment Score: {round(row['sentiment_score'] if row['sentiment'] == 'positive' else 0, 4)}")
        print(f"Negative Sentiment Score: {round(row['sentiment_score'] if row['sentiment'] == 'negative' else 0, 4)}")
        print(f"Neutral Sentiment Score: {round(row['sentiment_score'] if row['sentiment'] == 'neutral' else 0, 4)}")
        print("------------------------------------------------------------")

# Sample news data
news_data = [
    {"headline": "Global financial markets face sharp declines amid economic uncertainty", "sentiment": "negative", "impact_factor": 0.9},
    {"headline": "Tech industry achieves unprecedented success despite ongoing challenges", "sentiment": "positive", "impact_factor": 0.8},
    {"headline": "Government's decision to hike interest rates sparks economic debate", "sentiment": "neutral", "impact_factor": 0.7},
    {"headline": "Digital currencies experience rapid growth following government endorsement", "sentiment": "positive", "impact_factor": 0.85},
    {"headline": "Global oil markets experience significant price drops due to geopolitical tensions", "sentiment": "negative", "impact_factor": 0.95}
]

# Call the function with the news data
process_news_data(news_data)
