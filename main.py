import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from util import SmallCNN, explore_dataset, train_model, get_cam

# Define the dataset directory
data_dir = 'data/train'

# Explore the dataset
categories, category_distribution = explore_dataset(data_dir)
print("Categories:", categories)
print("Category distribution:", category_distribution)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SmallCNN(num_classes=len(categories))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# Load and preprocess an image for CAM visualization
img_path = 'data/test/some_category/example_image.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Get the predicted class
with torch.no_grad():
    output = model(img_tensor)
    _, predicted_class = torch.max(output, 1)

# Generate CAM
print("Generating Class Activation Map...")
cam = get_cam(model, img_tensor, predicted_class.item())

# Visualize the original image and CAM
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title('Class Activation Map')
plt.axis('off')

plt.tight_layout()
plt.show()
