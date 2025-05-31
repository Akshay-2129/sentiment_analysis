import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')


def analyze_sentiment(text):
    # Initialize VADER sentiment intensity analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Analyze sentiment of the text
    sentiment_scores = analyzer.polarity_scores(text)

    # Interpret sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    percentage = sentiment_scores['compound'] * 100  # Assuming compound score is used for overall sentiment

    return sentiment, percentage
