from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        me = request.form['message']
        # Convert input to float, skipping non-numeric values, and take only the first 30 values
        message = [float(x) for x in me.split() if x.replace('.', '').replace('-', '').isdigit()][:30]
        
        # Ensure we have exactly 30 features
        if len(message) != 30:
            return render_template('result.html', prediction="Error: Please provide exactly 30 numeric values.")
        
        vect = np.array(message).reshape(1, -1)
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)