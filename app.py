import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image
import numpy as np

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

BLOOD_GROUPS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
model_path = "best_model1.pth"
num_classes = len(BLOOD_GROUPS)
model = load_model(model_path, num_classes)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    if image is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        conf_score, predicted = torch.max(probabilities, 1)

    return {
        "Blood Group": BLOOD_GROUPS[predicted.item()],
        "Confidence": f"{conf_score.item()*100:.2f}%"
    }

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.main-div {
    background-color: #f7f7f7;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.output-div {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
        # 🩸 Blood Group Classification
        Upload an image of a blood sample to determine the blood group classification.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Blood Sample Image",
                type="numpy",
                sources="upload"
            )
            submit_btn = gr.Button(
                "Analyze Blood Group",
                variant="primary"
            )

        with gr.Column():
            output_label = gr.JSON(
                label="Prediction Results"
            )

    gr.Markdown(
        """
        ### How to use:
        1. Upload a clear image of a fingerprint
        2. Click 'Analyze Blood Group' to get the prediction
        3. View the predicted blood group and confidence score

        ### Note:
        - For best results, use high-quality images
        - Ensure proper lighting and focus
        - Images should be of individual blood samples
        """
    )

    submit_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=output_label
    )

demo.launch(share=True)
