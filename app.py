from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model= pickle.load(open('logistic_model_crimes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    crime_type = [int(x) for x in request.form.values()]
    final = np.array(crime_type).reshape(-1, 1)
    prediction = model.predict(final)[0]
    if prediction == 0:
        prediction = 'No arrest'
    else:
        prediction = "Yes, it will lead to arrest"

    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))
if  __name__== '__main__':
    app.run(port=8081, debug=True)

