from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

def ValuePredictor(weight_predict):
    to_predict = np.array(weight_predict).reshape(-1,1)
    model = joblib.load("model-development/Weight-prediction-using-linier-regression.pkl")
    result = model.predict(to_predict)
    return result[0]

@app.route("/", methods=['GET', 'POST'])
# def weight_prediction():
#     if request.method == 'GET':
#         return render_template("index.html")
#     elif request.method == 'POST':
#         #result = request.form
            
#         # print(dict(request.form))
#         weight_features = dict(request.form).values()
#         weight_features = np.array([float(x) for x in weight_features])
#         weight_features = weight_features.reshape(-1,1)
        

#         model = joblib.load("model-development/Weight-prediction-using-linier-regression.pkl")
#         # # #weight_features = std_scaler.transform([weight_features])
#         print(weight_features)
#         result = model.predict(weight_features)
#         # # # iris = {
#         # # #     '0': 'Iris Setosa',
#         # # #     '1': 'Iris Versicolor',
#         # # #     '2': 'Iris Virginica'
#         # # # }
#         # #result = iris[str(result[0])]
#         return render_template('index.html', result=result)
#     else:
#         return "Unsupported Request Method"


def result():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        weight_predict = request.form.to_dict()
        weight_predict = list(weight_predict.values())
        weight_predict = list(map(float, weight_predict))
        print(weight_predict)
        result = round(float(ValuePredictor(weight_predict)),2)

        
        
        
        # #result = request.form
            
        # # print(dict(request.form))
        # weight_features = dict(request.form).values()
        # weight_features = np.array([float(x) for x in weight_features])
        # weight_features = weight_features.reshape(-1,1)
        

        # model = joblib.load("model-development/Weight-prediction-using-linier-regression.pkl")
        # # # #weight_features = std_scaler.transform([weight_features])
        # print(weight_features)
        # result = model.predict(weight_features)
        # # # # iris = {
        # # # #     '0': 'Iris Setosa',
        # # # #     '1': 'Iris Versicolor',
        # # # #     '2': 'Iris Virginica'
        # # # # }
        # # #result = iris[str(result[0])]
        return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)