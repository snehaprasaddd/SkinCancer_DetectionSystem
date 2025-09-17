import torch
from torchvision import transforms, models
from PIL import Image
import json
import gradio as gr

# Paths to your trained model and label map JSON
MODEL_PATH = "best_skin_efnet_model.pth"
LABEL_MAP_PATH = "label_map.json"

# Load label map and create index-to-class dictionary
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

# Load model and adapt the classifier for the number of classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)  # Updated for newer torchvision
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(idx_to_label))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Preprocessing (should match your training/validation pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def infer(image):
    image = image.convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
    predicted_label = idx_to_label[pred.item()]
    return f"{predicted_label} (confidence: {conf.item()*100:.2f}%)"

# Gradio Web Interface
gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Upload a skin lesion image"),
    outputs=gr.Textbox(label="Predicted Disease & Confidence"),
    title="Skin Disease Classifier"
).launch()
