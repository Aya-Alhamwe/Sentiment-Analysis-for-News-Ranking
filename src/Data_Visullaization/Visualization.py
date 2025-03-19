import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Plotting the distribution of sentiment labels using a count plot
def plot_sentiment_distribution(news_df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=news_df['sentiment'].astype(str), palette='coolwarm')
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

# Analyzing the relationship between news length and sentiment using a box plot
def plot_news_length_vs_sentiment(news_df):
    news_df['news_length'] = news_df['news'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=news_df['sentiment'], y=news_df['news_length'], palette='coolwarm')
    plt.title('Relationship between News Length and Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('News Length')
    plt.show()

# Create WordCloud for Negative News
def create_negative_wordcloud(news_df):
    negative_text = " ".join(news_df[news_df['sentiment'] == 'NEGATIVE']['news'].dropna())
    plt.figure(figsize=(10, 5))
    wordcloud_negative = WordCloud(background_color='black', colormap='Reds', max_words=100).generate(negative_text)
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Common Words in Negative News")
    plt.show()

# Create WordCloud for Positive News
def create_positive_wordcloud(news_df):
    positive_text = " ".join(news_df[news_df['sentiment'] == 'POSITIVE']['news'].dropna())
    plt.figure(figsize=(10, 5))
    wordcloud_positive = WordCloud(background_color='black', colormap='Blues', max_words=100).generate(positive_text)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis("off")
    plt.title("Most Common Words in Positive News")
    plt.show()

# Drawing the relationship between headline length and sentiment
def plot_headline_length_vs_sentiment(news_df):
    news_df['headline_length'] = news_df['news'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=news_df['sentiment'], y=news_df['headline_length'], palette='magma')
    plt.title('Impact of Headline Length on Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Headline Length')
    plt.show()

# Converting the 'date' column to datetime format and extracting day of the week
def plot_impact_by_day_of_week(news_df):
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['day_of_week'] = news_df['date'].dt.dayofweek
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=news_df['day_of_week'], y=news_df['compound'], palette='viridis')
    plt.title('News Impact Score by Day of the Week')
    plt.xlabel('Day of the Week (0 = Monday)')
    plt.ylabel('Impact Score')
    plt.show()

# Calculating the correlation between sentiment scores and plotting a heatmap
def plot_sentiment_correlation(news_df):
    correlation = news_df[['neg', 'neu', 'pos', 'compound']].corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Sentiment Scores')
    plt.show()

# Calculate average impact by day and plot as a bar chart
def plot_average_impact_by_day(news_df):
    daily_impact = news_df.groupby('day_of_week')['compound'].mean()
    plt.figure(figsize=(8, 5))
    daily_impact.plot(kind='bar', color='skyblue')
    plt.title('Average Compound Score by Day of the Week')
    plt.xlabel('Day of the Week (0 = Monday)')
    plt.ylabel('Average Compound Score')
    plt.show()

# Plotting the distribution of sentiment labels as a pie chart
def plot_sentiment_pie_chart(news_df):
    sentiment_counts = news_df['sentiment'].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'gray', 'green'])
    plt.title('Sentiment Distribution')  
    plt.ylabel('')  
    plt.show()

# Calculate total compound score by sentiment and plot as a bar chart
def plot_total_compound_by_sentiment(news_df):
    total_impact = news_df.groupby('sentiment')['compound'].sum()
    total_impact.plot(kind='bar', color='lightcoral')
    plt.title('Total Compound Score by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Total Compound Score')
    plt.show()

# Sentiment analysis by day and plot trend over time
def plot_sentiment_over_time(news_df):
    daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].value_counts().unstack().fillna(0)
    daily_sentiment.plot(kind='line', figsize=(12, 6))
    plt.title('Sentiment Distribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment', labels=['Positive', 'Neutral', 'Negative'])
    plt.show()

# Drawing the relationship between the length of the news and the feelings
def plot_news_length_vs_sentiment_score(news_df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=news_df['news_length'], y=news_df['compound'], hue=news_df['sentiment'], palette='coolwarm')
    plt.title('Effect of News Length on Sentiment Classification')
    plt.xlabel('News Length')
    plt.ylabel('Compound Score')
    plt.show()
