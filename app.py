import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

st.title("Offshore Pipeline Corrosion Detector")
st.write("Upload an image of a pipeline to check for corrosion levels.")

# Load Model Function
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    # Ensure corrosion_model.pth is in the same folder
    model.load_state_dict(torch.load('corrosion_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ['Healthy', 'Mild Corrosion', 'Severe Corrosion']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        status = classes[predicted.item()]
        
    st.subheader(f"Prediction: {status}")