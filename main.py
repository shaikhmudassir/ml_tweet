from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import model

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        polarity = model.analyze_sentiment(text)
        return render_template('result.html', polarity=polarity)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
