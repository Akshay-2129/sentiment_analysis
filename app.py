from flask import Flask, request, render_template
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)

# Function to analyze sentiment of text
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Interpret sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    percentage = sentiment_scores['compound'] * 100  # Compound score as the overall sentiment
    return sentiment, percentage

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        results = []
        sentiment_distribution = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        # Handle only text input
        user_texts = request.form.get('user_text')
        if user_texts:
            texts = user_texts.split('\n')
            for text in texts:
                sentiment, percentage = analyze_sentiment(text)
                results.append((text, sentiment, percentage))
                sentiment_distribution[sentiment] += 1

        return render_template('result.html', results=results, sentiment_data=sentiment_distribution)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
