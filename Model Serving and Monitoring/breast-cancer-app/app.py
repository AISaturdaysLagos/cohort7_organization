from flask import Flask, jsonify, request
import traceback
import joblib
import numpy as np

app = Flask(__name__)
rf = joblib.load("breast_cancer_model.sav")  # loads model"

@app.route('/predict', methods=['POST'])  # API endpoint URL would consist /predict
def inference():
    if rf:
        try:
            # gets input data and confirms that it is in the right format.
            json_ = request.get_json(force=True) 
            query = json_ if type(json_[0])==list else [json_] 
            # makes prediction amd decodes labels
            status = {1: 'Positive', 0: "Negative"} 
            prediction = [status[i] for i in rf.predict(query).tolist()]

            return jsonify(predictions=prediction)

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('Model is not avaliable.')

if __name__ == '__main__':
    app.run(port=5000, debug=False)