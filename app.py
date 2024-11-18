import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import gdown
import os

st.title("Pet Breed Classifier")

# Скачивание модели из Google Drive, если она отсутствует
MODEL_URL = "https://drive.google.com/uc?id=12g8KcEZLD-7Y8ExbIStZNjTjEmu0TUXR"  # Замените на свою ссылку
MODEL_PATH = "improved_model.pth"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model downloaded successfully!")

# Загрузка модели
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))  # Загрузка модели для CPU
model.eval()

# Интерфейс для загрузки изображения
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
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
        st.write(f"Predicted Breed: {predicted.item()}")
