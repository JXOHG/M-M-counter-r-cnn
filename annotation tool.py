import cv2
import numpy as np
import json
import os
from pathlib import Path

class AnnotationTool:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.current_image = None
        self.current_image_name = ""
        self.current_boxes = []  # Will store [x1, y1, x2, y2, color_id]
        self.current_color = 1
        self.drawing = False
        self.start_point = None
        self.all_annotations = {}
        self.scale_factor = 1.0
        self.target_height = 800  # Maximum window height
        self.target_width = 1200  # Maximum window width
        
        # Define M&M colors and their IDs
        self.colors = {
            1: ("Red", (35, 36, 138)),
            2: ("Blue", (135, 72, 0)),
            3: ("Green", (43, 117, 60)),
            4: ("Yellow", (0, 179, 226)),
            5: ("Brown", (18, 23, 29)),
            6: ("Orange", (30, 98, 229))
        }
        
        # Load all images from folder
        self.load_images()
        self.current_idx = 0
        if self.images:
            self.load_current_image()

    def resize_image(self, image):
        """Resize image to fit screen while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        h_scale = self.target_height / height
        w_scale = self.target_width / width
        self.scale_factor = min(h_scale, w_scale)
        
        if self.scale_factor < 1:  # Only resize if image is too large
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            return cv2.resize(image, (new_width, new_height))
        else:
            self.scale_factor = 1.0
            return image

    def scale_to_original(self, x, y):
        """Convert display coordinates to original image coordinates"""
        return int(x / self.scale_factor), int(y / self.scale_factor)

    def scale_to_display(self, x, y):
        """Convert original image coordinates to display coordinates"""
        return int(x * self.scale_factor), int(y * self.scale_factor)
    
    def load_images(self):
        """Load all images from the specified folder"""
        valid_extensions = ['.jpg', '.jpeg', '.png']
        self.images = [f for f in os.listdir(self.image_folder) 
                      if any(f.lower().endswith(ext) for ext in valid_extensions)]
        self.images.sort()
    
    def load_current_image(self):
        """Load the current image and any existing annotations"""
        if 0 <= self.current_idx < len(self.images):
            self.current_image_name = self.images[self.current_idx]
            img_path = os.path.join(self.image_folder, self.current_image_name)
            original_image = cv2.imread(img_path)
            self.current_image = self.resize_image(original_image)
            self.current_boxes = []
            
            # Load existing annotations if any
            if self.current_image_name in self.all_annotations:
                self.current_boxes = self.all_annotations[self.current_image_name]['boxes']
    
    def draw_color_selector(self, img):
        """Draw color selector menu at the top of the image"""
        padding = 10
        for color_id, (color_name, bgr) in self.colors.items():
            # Draw color box
            cv2.rectangle(img, (padding, 10), (padding + 30, 40), bgr, -1)
            # Add color number
            cv2.putText(img, str(color_id), (padding + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Highlight selected color
            if color_id == self.current_color:
                cv2.rectangle(img, (padding - 2, 8), (padding + 32, 42), (255, 255, 255), 2)
            padding += 50
        
        # Display current color name
        cv2.putText(img, f"Selected: {self.colors[self.current_color][0]}", 
                   (padding + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.current_image.copy()
            # Draw all existing boxes
            self.draw_boxes(img_copy)
            # Draw current box
            cv2.rectangle(img_copy, self.start_point, (x, y), 
                         self.colors[self.current_color][1], 2)
            self.draw_color_selector(img_copy)
            cv2.imshow('Annotation Tool', img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Convert display coordinates to original image coordinates
            x1, y1 = self.scale_to_original(*self.start_point)
            x2, y2 = self.scale_to_original(x, y)
            
            # Ensure coordinates are in correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Add new box with original image coordinates
            self.current_boxes.append([x1, y1, x2, y2, self.current_color])
            
            # Update display
            img_copy = self.current_image.copy()
            self.draw_boxes(img_copy)
            self.draw_color_selector(img_copy)
            cv2.imshow('Annotation Tool', img_copy)

    def draw_boxes(self, img):
        """Draw all boxes on the image"""
        for box in self.current_boxes:
            # Convert original coordinates to display coordinates
            x1, y1 = self.scale_to_display(box[0], box[1])
            x2, y2 = self.scale_to_display(box[2], box[3])
            color_id = box[4]
            cv2.rectangle(img, (x1, y1), (x2, y2), self.colors[color_id][1], 2)
    
    def run(self):
        """Run the annotation tool"""
        if not self.images:
            print("No images found in the specified folder!")
            return
        
        print("\nControls:")
        print("1-6: Select M&M color")
        print("n: Next image")
        print("p: Previous image")
        print("u: Undo last box")
        print("s: Save annotations")
        print("q: Quit and save")
        
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        while True:
            img_copy = self.current_image.copy()
            self.draw_boxes(img_copy)
            self.draw_color_selector(img_copy)
            cv2.imshow('Annotation Tool', img_copy)
            key = cv2.waitKey(1) & 0xFF
            
            # Color selection
            if ord('1') <= key <= ord('6'):
                self.current_color = key - ord('0')
            
            elif key == ord('n'):  # Next image
                self.save_current_annotations()
                self.current_idx = min(self.current_idx + 1, len(self.images) - 1)
                self.load_current_image()
            
            elif key == ord('p'):  # Previous image
                self.save_current_annotations()
                self.current_idx = max(self.current_idx - 1, 0)
                self.load_current_image()
            
            elif key == ord('u'):  # Undo last box
                if self.current_boxes:
                    self.current_boxes.pop()
            
            elif key == ord('s'):  # Save annotations
                self.save_current_annotations()
                self.save_all_annotations()
                print("Annotations saved!")
            
            elif key == ord('q'):  # Quit
                self.save_current_annotations()
                self.save_all_annotations()
                break
        
        cv2.destroyAllWindows()
    
    def save_current_annotations(self):
        """Save annotations for current image"""
        # Save original coordinates
        boxes = [[box[0], box[1], box[2], box[3]] for box in self.current_boxes]
        labels = [box[4] for box in self.current_boxes]
        self.all_annotations[self.current_image_name] = {
            'boxes': boxes,
            'labels': labels
        }
    
    def save_all_annotations(self):
        """Save all annotations to a JSON file"""
        output_path = os.path.join(self.image_folder, 'annotations.json')
        with open(output_path, 'w') as f:
            json.dump(self.all_annotations, f, indent=4)
    
    def get_training_format(self):
        """Convert annotations to training format"""
        img_paths = []
        annotations = []
        
        for img_name, anno in self.all_annotations.items():
            img_paths.append(os.path.join(self.image_folder, img_name))
            annotations.append({
                'boxes': anno['boxes'],
                'labels': anno['labels']
            })
        
        return img_paths, annotations

def main():
    """Main function to run the annotation tool"""
    folder_path = input("Enter the folder path containing your M&M images: ")
    tool = AnnotationTool(folder_path)
    tool.run()
    
    # Get annotations in training format
    img_paths, annotations = tool.get_training_format()
    print(f"\nAnnotated {len(img_paths)} images")
    
    # Print color statistics
    color_counts = {}
    for anno in annotations:
        for label in anno['labels']:
            color_counts[label] = color_counts.get(label, 0) + 1
    
    print("\nM&M counts by color:")
    for color_id, count in color_counts.items():
        color_name = tool.colors[color_id][0]
        print(f"{color_name}: {count}")

if __name__ == "__main__":
    main()