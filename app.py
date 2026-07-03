import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

st.title("Steel Surface Defect Classification")
st.write("Upload a steel surface image to classify the defect type using ResNet50.")

class_names = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "normal",
    "bent",
    "broken",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\ravi.dalal\Desktop\Python\Projects\Defect_Classification\resnet50_neu_finetuned.pth"

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    print("Checkpoint FC weight shape:", checkpoint["fc.weight"].shape)

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload steel surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability (%)": [round(p.item() * 100, 2) for p in probabilities]
    })

    prob_df = prob_df.sort_values(by="Probability (%)", ascending=False)

    st.subheader("Class-wise Probability")
    st.dataframe(prob_df, use_container_width=True)
    st.bar_chart(prob_df.set_index("Class"))