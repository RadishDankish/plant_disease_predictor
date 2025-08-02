from flask import Flask, request, jsonify, render_template # Import render_template
import torch
from torchvision import transforms
from PIL import Image
import io
from torch.utils.data import DataLoader
from torchvision import datasets

# Define model
from model import CNN  # your CNN model class

app = Flask(__name__, template_folder='templates') # Specify template folder

# Load model
model = CNN(38)
model.load_state_dict(torch.load("my_model.pth", map_location=torch.device('cpu')))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()

])

train_dataset = datasets.ImageFolder(root='/home/danish/Desktop/ML_DL/plant_disease/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train', transform=transform)
class_names = train_dataset.classes

@app.route('/') # New route for the homepage
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)