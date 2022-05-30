# Import libraries
from flask import Flask, request, jsonify
from keras.models import model_from_json
from keras.models import load_model
import pickle

app = Flask(__name__)

# Load the model
model = load_model('mmy_best_model.hdf5')

@app.route('/api/',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
