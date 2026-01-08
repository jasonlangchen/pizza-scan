from typing import cast
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict_topping(image_path, model_path, class_names):
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()

  img = Image.open(image_path).convert('RGB')
  img_tensor = cast(torch.Tensor, transform(img))
  input_batch = img_tensor.unsqueeze(0).to(device)

  with torch.no_grad():
    outputs = model(input_batch)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)

  topping = class_names[predicted_idx.item()]
  print(f"Prediction: {topping} ({confidence.item()*100:.2f}% confidence)")

my_classes = ['basil','mushroom','pepper','pepperoni', 'pineapple', 'sausage'] 

predict_topping(
  image_path='test_pizza.jpg', 
  model_path='best_model.pth', 
  class_names=my_classes
)