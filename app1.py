import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import json

st.title("Skin Disease Classifier")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("label_map.json") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(idx_to_label))
    model.load_state_dict(torch.load("best_skin_efnet_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, idx_to_label, device

def predict(image, model, idx_to_label, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = image.convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    label = idx_to_label[pred.item()]
    return label, confidence.item()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    model, idx_to_label, device = load_model_and_labels()
    label, conf = predict(image, model, idx_to_label, device)
    
    st.success(f"Prediction: **{label}**")
    st.info(f"Confidence: **{conf*100:.2f}%**")
