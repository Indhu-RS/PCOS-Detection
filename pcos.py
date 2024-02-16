from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('model.h5')

# Define function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define function to get prediction
def get_prediction(image):
    prediction = model.predict(image)
    return prediction

# Define function to get class label
def get_class_label(prediction):
    if prediction[0][0] > prediction[0][1]:
        return "infected"
    else:
        return "not infected"

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the file to a secure location
            filename = secure_filename(file.filename)
            file_path = 'uploads/' + filename
            file.save(file_path)
            # Preprocess the uploaded image
            image = preprocess_image(file_path)
            # Get prediction
            prediction = get_prediction(image)
            # Get class label
            class_label = get_class_label(prediction)
            return render_template('result.html', image_file=file_path, class_label=class_label)

if __name__ == '__main__':
    app.run(debug=True)
