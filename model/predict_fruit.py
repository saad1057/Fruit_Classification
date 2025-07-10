import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class_names = [
    "apple_fruit",
    "banana_fruit",
    "grapes_fruit",
    "kiwi_fruit",
    "mango_fruit",
    "orange_fruit",
    "peach_fruit",
    "pineapple_fruit",
    "strawberry_fruit",
    "watermelon_fruit"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, len(class_names))
)
model.load_state_dict(torch.load("checkpoints/resnet18_fruit.pth", map_location=device))
model = model.to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

if __name__ == "__main__":
    test_image_path = "test_images/apple2.jpg"  
    if os.path.exists(test_image_path):
        prediction = predict_image(test_image_path)
        print(f"Predicted Fruit: {prediction}")
    else:
        print("Image path not found.")
