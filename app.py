import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import gdown
import os
import torch.nn as nn

st.title("Pet Breed Classifier")

# Ссылка на Google Drive для загрузки модели
MODEL_URL = "https://drive.google.com/uc?id=12g8KcEZLD-7Y8ExbIStZNjTjEmu0TUXR"
MODEL_PATH = "improved_model.pth"

# Проверка наличия модели и её загрузка
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model downloaded successfully!")

# Определение архитектуры модели (пример, замените на вашу)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=23):  # Укажите количество классов
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 112 * 112, num_classes)  # Размеры под архитектуру

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Загрузка модели
try:
    model = SimpleCNN(num_classes=23)  # Укажите количество классов
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Список классов (пород) из вашего датасета
CLASSES = [
    'abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog',
    'chihuahua', 'corgi', 'dachshund', 'german shepherd', 'golden retriever',
    'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
    'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu',
    'siamese cat', 'sphynx', 'yorkshire terrier'
]

# Интерфейс для загрузки изображения
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Преобразование изображения для модели
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = CLASSES[predicted.item()]  # Название класса из списка CLASSES
        st.write(f"Predicted Breed: **{predicted_class}**")
