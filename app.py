import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.title("Pet Breed Classifier")

model = torch.load("improved_model.pth")  # Загрузите вашу обученную модель
model.eval()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        st.write(f"Predicted Breed: {predicted.item()}")
