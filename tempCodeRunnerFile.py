from flask import Flask, request, render_template
from textsentimentanalysis import analyze_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_texts = request.form.get('user_text')
        if not user_texts:
            return render_template('index.html', error="Please enter some text for analysis.")
        
        texts = user_texts.split('\n')
        results = []
        for text in texts:
            sentiment, percentage = analyze_sentiment(text)
            results.append((text, sentiment, percentage))

        # Calculate sentiment distribution
        sentiment_distribution = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for result in results:
            sentiment_distribution[result[1]] += 1

        return render_template('result.html', results=results, sentiment_data=sentiment_distribution)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
