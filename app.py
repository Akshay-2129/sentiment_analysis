from flask import Flask, request, render_template
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)

# Set upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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

        # Handle text input
        user_texts = request.form.get('user_text')
        if user_texts:
            texts = user_texts.split('\n')
            for text in texts:
                sentiment, percentage = analyze_sentiment(text)
                results.append((text, sentiment, percentage))
                sentiment_distribution[sentiment] += 1

        # Handle file input
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                # Read file and process each line
                with open(filename, 'r') as f:
                    file_content = f.read()
                    texts = file_content.split('\n')
                    for text in texts:
                        sentiment, percentage = analyze_sentiment(text)
                        results.append((text, sentiment, percentage))
                        sentiment_distribution[sentiment] += 1

        return render_template('result.html', results=results, sentiment_data=sentiment_distribution)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
