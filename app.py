import numpy as np
from flask import Flask, render_template, request
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    float_val = [float(x) for x in request.form.values()]
    final_val = np.array(float_val).reshape(1,6)
    prediction = model.predict(final_val)

    output = round(prediction[0],2)

    return render_template('index.html',result="The Insurance charges would be $ {}".format(output))

if __name__=="__main__":
    app.run(debug=True)