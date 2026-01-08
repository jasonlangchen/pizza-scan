import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Data transformations
train_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomRotation(15),
  transforms.ColorJitter(brightness=0.2),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset_full = datasets.ImageFolder(root='pizza_data', transform=train_transform)
val_dataset_full = datasets.ImageFolder(root='pizza_data', transform=val_transform)

# Stratified K-fold
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

labels = train_dataset_full.targets

# Cross validation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_accuracy = 0.0
save_path = "best_model.pth"

for fold, (train_ids, val_ids) in enumerate(skf.split(np.arange(len(labels)), labels)):
  print(f'Fold {fold}')

  # Reinitialize model and optimizer
  model = models.resnet18(weights="DEFAULT")
  model.fc = nn.Linear(model.fc.in_features, len(train_dataset_full.classes))
  model.to(device)
  
  optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
  criterion = nn.CrossEntropyLoss()

  # Subsets for training and validation
  train_subset = Subset(train_dataset_full, train_ids.tolist())
  val_subset = Subset(val_dataset_full, val_ids.tolist())

  # Create data loaders
  train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

  for epoch in range(10):
    # Training loop
    model.train()
    running_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f'Fold {fold} - Epoch {epoch+1}/10 [Train]')
    for inputs, targets in train_pbar:
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      
      train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_subset)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    
    val_pbar = tqdm(val_loader, desc=f'Fold {fold} - Epoch {epoch+1}/10 [Val]')
    with torch.no_grad():
      for inputs, targets in val_pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        val_pbar.set_postfix({'acc': f'{correct/total:.4f}'})

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')

    # Save the best model
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      torch.save(model.state_dict(), save_path)
      print(f'Best model saved with accuracy: {best_accuracy:.4f}')