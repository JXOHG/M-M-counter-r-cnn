import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

class MMPredictor:
    def __init__(self, model_path):
        # Color mapping
        self.colors = {
            1: ("Red", (255, 0, 0)),
            2: ("Blue", (0, 0, 255)),
            3: ("Green", (0, 255, 0)),
            4: ("Yellow", (255, 255, 0)),
            5: ("Brown", (165, 42, 42)),
            6: ("Orange", (255, 165, 0))
        }
        
        # Set up device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        if self.model is None:
            raise ValueError("Failed to load the model. Please check the error messages above.")
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        try:
            # Initialize model with same architecture
            model = fasterrcnn_resnet50_fpn(pretrained=False)
            
            # Modify for number of classes (background + 6 colors)
            num_classes = 7
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes)
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            return model
        except PermissionError:
            print(f"Error: Permission denied when trying to load the model from {model_path}")
            print("Please check if you have the necessary permissions to access this file.")
            return None
        except FileNotFoundError:
            print(f"Error: The model file {model_path} was not found.")
            print("Please make sure you've provided the correct path to the model file.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {str(e)}")
            return None

    def predict_image(self, image_path, confidence_threshold=0.5, save_output=True):
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        img_tensor = F.to_tensor(image)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model([img_tensor.to(self.device)])
        
        # Get prediction details
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        
        # Convert image for display
        img_display = np.array(image)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(img_display)
        
        # Count M&Ms by color
        color_counts = {}
        
        # Draw predictions
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score > confidence_threshold:
                # Update color count
                color_name = self.colors[label][0]
                color_counts[color_name] = color_counts.get(color_name, 0) + 1
                
                # Draw box
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, color=np.array(self.colors[label][1])/255, linewidth=2
                ))
                
                # Add label
                plt.text(
                    x1, y1-5, 
                    f"{self.colors[label][0]}: {score:.2f}",
                    bbox=dict(facecolor='white', alpha=0.8)
                )
        
        # Add title with counts
        title = "Detected M&Ms:\n"
        for color, count in color_counts.items():
            title += f"{color}: {count}, "
        plt.title(title[:-2])  # Remove last comma and space
        
        plt.axis('off')
        
        # Save or show result
        if save_output:
            output_path = os.path.join(
                os.path.dirname(image_path),
                f"detected_{os.path.basename(image_path)}"
            )
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Saved detection result to: {output_path}")
        
        plt.show()
        
        return color_counts

def main():
    # Get paths
    model_path = input("Enter path to trained model (mm_detector.pth): ")
    if not model_path:
        model_path = "mm_detector.pth"
    
    image_path = input("Enter path to image for prediction: ")
    
    try:
        # Create predictor
        predictor = MMPredictor(model_path)
        
        # Make prediction
        print("\nMaking prediction...")
        color_counts = predictor.predict_image(image_path)
        
        # Print results
        print("\nDetection Results:")
        total_mms = 0
        for color, count in color_counts.items():
            print(f"{color} M&Ms: {count}")
            total_mms += count
        print(f"\nTotal M&Ms detected: {total_mms}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except FileNotFoundError:
        print(f"Error: The image file {image_path} was not found.")
        print("Please make sure you've provided the correct path to the image file.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
