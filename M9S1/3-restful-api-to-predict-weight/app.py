from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(open('./model-development/Weight-prediction-using-linier-regression.pkl', 'rb'))
    
@app.route("/", methods=['GET', 'POST'])
def predict_weight():

	if request.method == 'POST':
		Height = request.form['Height']
		Height = float(Height)
		Gender = request.form['Gender']
		Gender = float(Gender)
		result = model.predict([[Height, Gender]])
		return render_template("index.html", result = result[0])
	return render_template("index.html")



if __name__ == '__main__':
    app.run(port=5000, debug=True)