# import torch
# from torchvision import transforms
# from PIL import Image
# import json

# # Load EMD (Esri Model Definition)
# with open('PlantLeafDiseaseClassification.emd', 'r') as f:
#     emd = json.load(f)

# # Load class names and image size
# class_names = [cls['Name'] for cls in emd['Classes']]
# img_height = emd['ImageHeight']
# img_width = emd['ImageWidth']

# # Load the model
# model = torch.load('PlantLeafDiseaseClassification.pth', map_location=torch.device('cpu'))
# model.eval()

# # Define preprocessing
# transform = transforms.Compose([
#     transforms.Resize((img_height, img_width)),
#     transforms.ToTensor()
# ])

# # Test image path
# image_path = 'test.jpeg'  # Replace with a sample image

# # Open and preprocess image
# img = Image.open(image_path).convert('RGB')
# img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# # Inference
# with torch.no_grad():
#     output = model(img_tensor)
#     predicted_index = torch.argmax(output, dim=1).item()
#     predicted_class = class_names[predicted_index]

# print(f"✅ Prediction: {predicted_class}")


import torch
from torchvision import models, transforms
from PIL import Image
import json

# Load EMD file
with open('PlantLeafDiseaseClassification.emd', 'r') as f:
    emd = json.load(f)

class_names = [cls['Name'] for cls in emd['Classes']]
img_height = emd['ImageHeight']
img_width = emd['ImageWidth']
num_classes = len(class_names)

# STEP 1: Define the model architecture
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# STEP 2: Load the saved weights (state_dict)
model.load_state_dict(torch.load('PlantLeafDiseaseClassification.pth', map_location='cpu'))
model.eval()

# STEP 3: Define image transforms
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

# STEP 4: Load and transform your test image
image_path = 'test.jpg'  # Replace with your own image path
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# STEP 5: Predict
with torch.no_grad():
    outputs = model(img_tensor)
    pred_idx = torch.argmax(outputs, dim=1).item()
    prediction = class_names[pred_idx]

print(f"✅ Prediction: {prediction}")
