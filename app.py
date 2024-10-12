from flask_bootstrap import Bootstrap
from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

gender_app = Flask(__name__)
Bootstrap(gender_app)

@gender_app.route('/')
def index():
    return render_template('index.html')

@gender_app.route('/predict', methods=['POST'])
def predict():
    # Load the dataset and model
    df = pd.read_csv("Names_dataset.csv")
    x_df = df.name
    y_df = df.gender
    corpus = x_df.values.astype('U')
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) 

    # Load the Naive Bayes model
    nb = open("naivebayes.pkl", "rb")
    clf_1 = joblib.load(nb)

    # Receive input and handle empty name submission
    if request.method == 'POST':
        name_query = request.form.get('name_query')  # Safely get form input
        if not name_query:  # Check if the input is empty
            error = "Please enter a name."
            return render_template('index.html', error=error)  # Render index with error

        # Proceed if a name was submitted
        data = [name_query]
        vct = cv.transform(data).toarray()
        my_prediction = clf_1.predict(vct)
        
        return render_template('results.html', prediction=my_prediction, name=name_query.upper())

if __name__ == '__main__':
    gender_app.run(debug=True)
