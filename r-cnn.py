import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

class ColoredMMDataset(Dataset):
    def __init__(self, image_folder, annotation_file):
        self.image_folder = image_folder
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Get list of image files that have annotations
        self.image_files = list(self.annotations.keys())
        
        # Color mapping for visualization
        self.colors = {
            1: "Red",
            2: "Blue",
            3: "Green",
            4: "Yellow",
            5: "Brown",
            6: "Orange"
        }

    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_filename)
        img = Image.open(img_path).convert("RGB")
        
        # Convert image to tensor
        img_tensor = F.to_tensor(img)
        
        # Get annotations for this image
        anno = self.annotations[img_filename]
        
        # Convert annotations to tensor format
        boxes = torch.as_tensor(anno['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(anno['labels'], dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return img_tensor, target

    def __len__(self):
        return len(self.image_files)

def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Modify the box predictor for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    
    return model

def train_model(model, data_loader, device, num_epochs=10):
    # Move model to device
    model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, targets in data_loader:
            # Move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i in range(min(num_samples, len(dataset))):
        # Get image and target
        img, target = dataset[i]
        
        # Make prediction
        with torch.no_grad():
            prediction = model([img.to(device)])
        
        # Convert image for display
        img_display = img.permute(1, 2, 0).cpu().numpy()
        
        # Plot original image with ground truth
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img_display)
        
        # Draw ground truth boxes
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.numpy()
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, color='green', linewidth=2))
            plt.text(x1, y1, dataset.colors[label.item()],
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Plot predictions
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(img_display)
        
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        
        # Only show predictions with score > 0.5
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > 0.5:
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                                fill=False, color='red', linewidth=2))
                plt.text(x1, y1, f"{dataset.colors[label]}\n{score:.2f}",
                        bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Predictions')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Get data paths
    image_folder = input("Enter the folder path containing your M&M images: ")
    annotation_file = os.path.join(image_folder, 'annotations.json')
    
    # Create dataset
    dataset = ColoredMMDataset(image_folder, annotation_file)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    num_classes = 7  # Background + 6 M&M colors
    model = get_model(num_classes)
    
    # Train model
    model = train_model(model, train_loader, device)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(image_folder, 'mm_detector.pth'))
    
    # Visualize some predictions
    print("\nVisualizing predictions...")
    visualize_predictions(model, val_dataset, device)
    
    print("\nTraining completed! Model saved as 'mm_detector.pth'")

if __name__ == "__main__":
    main()