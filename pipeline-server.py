# need to run pip install flask

from sklearn.externals import joblib
from flask import Flask, request
from jinja2 import Template

p = joblib.load('sentiment-model.pkl')

app = Flask(__name__)

def pred(text):
    return p.predict([text])[0]

@app.route('/')
def index():
    text = request.args.get('text')
    if text:
        prediction = pred(text)
    else:
        prediction = ""

    template = Template("""
    <html>
        <body>
            <h1>Scikit Model Server</h1>
            <form>
                <input type="text" name="text">
                <input type="submit" >
            </form>
            <p>Prediction: {{ prediction }}</p>
        </body>
    </html>
    """)
    return template.render(prediction=prediction)





if __name__ == '__main__':
    app.run(port=8000)
