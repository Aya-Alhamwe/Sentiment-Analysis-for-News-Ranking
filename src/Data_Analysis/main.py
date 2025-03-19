import pandas as pd

# Load dataset from the specified path
csv_path = "/kaggle/input/news-sentiment-analysis/news.csv"
news_df = pd.read_csv(csv_path)

# Display 6 random rows
print(news_df.sample(6))

# Show the shape and info of the dataset
print(news_df.shape)
news_df.info()

# Check for missing values and duplicates
print(news_df.isnull().sum())
print(news_df.duplicated().sum())

# Get the list of columns in the dataset
print(news_df.columns)

# Calculating sentiment distribution as percentages and counts
sentiment_counts = news_df['sentiment'].value_counts()
total = sentiment_counts.sum()

sentiment_distribution = pd.DataFrame({
    'sentiment': sentiment_counts.index,
    'norm_counts': sentiment_counts / total,
    'counts': sentiment_counts
})

print(sentiment_distribution)

# Selecting and printing the 30th news article from the dataset
exampleNews = news_df['news'][30]
print(exampleNews)

# Calculate the total positive and negative compound scores and print the results
positive_impact = news_df[news_df['sentiment'] == 'POSITIVE']['compound'].sum()
negative_impact = news_df[news_df['sentiment'] == 'NEGATIVE']['compound'].sum()

print(f"Total Positive Impact: {positive_impact}")
print(f"Total Negative Impact: {negative_impact}")
