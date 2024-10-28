from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

app = Flask(__name__)
model = load_model('model_casia_run1.h5')

def prepare_image(image_path):
    image_size = (128, 128)
    quality = 90
    image = Image.open(image_path).convert('RGB')
    temp_filename = 'temp_file_name.jpg'
    image.save(temp_filename, 'JPEG', quality=quality) 
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return np.array(ela_image.resize(image_size)).reshape(-1, 128, 128, 3) / 255.0
def predict_real_fake(image_path):
    image = prepare_image(image_path)
    predictions = model.predict(image)
    real_percentage = predictions[0][1] * 100
    fake_percentage = predictions[0][0] * 100
    return real_percentage, fake_percentage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    image_path = 'uploaded_image.jpg'
    file.save(image_path)
    
    real_percent, fake_percent = predict_real_fake(image_path)
    return jsonify({'real_percent': real_percent, 'fake_percent': fake_percent})

if __name__ == '__main__':
    app.run(debug=True)
