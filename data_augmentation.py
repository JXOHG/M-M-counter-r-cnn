import cv2
import numpy as np
from typing import List, Tuple
import os
import logging
from scipy.ndimage import rotate
import albumentations as A

class DataAugmentation:
    """
    Handles image augmentation for training data generation.
    """
    
    def __init__(self, output_dir: str = "augmented_data"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.GaussNoise(p=0.6),
            A.RandomGamma(p=0.6),
            A.Rotate(limit=45, p=0.7),
            A.RandomScale(scale_limit=0.2, p=0.7),
            A.Perspective(p=0.5),
        ])
        
    def augment_image(self, image: np.ndarray, num_augmentations: int = 10) -> List[np.ndarray]:
        """
        Generate augmented versions of input image.
        
        Args:
            image: Input image
            num_augmentations: Number of augmented images to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        try:
            for i in range(num_augmentations):
                # Apply augmentation pipeline
                augmented = self.transform(image=image)['image']
                augmented_images.append(augmented)
                
                # Save augmented image
                output_path = os.path.join(self.output_dir, f"aug_{i}.jpg")
                cv2.imwrite(output_path, augmented)
                
            return augmented_images
            
        except Exception as e:
            self.logger.error(f"Error in image augmentation: {str(e)}")
            raise
            
    def generate_kernel_variations(self, base_kernel_size: Tuple[int, int], 
                                   num_variations: int = 5) -> List[Tuple[int, int]]:
        """
        Generate variations of kernel sizes for training.
        
        Args:
            base_kernel_size: Base kernel size
            num_variations: Number of variations to generate
            
        Returns:
            List of kernel size tuples
        """
        kernel_sizes = []
        base_size = base_kernel_size[0]  # Assuming square kernel
        
        try:
            for i in range(-num_variations, num_variations + 1):
                new_size = max(3, base_size + i * 2)  # Ensure odd numbers
                kernel_sizes.append((new_size, new_size))
                
            return kernel_sizes
            
        except Exception as e:
            self.logger.error(f"Error generating kernel variations: {str(e)}")
            raise

    def generate_tile_grid_variations(self, base_tile_grid_size: Tuple[int, int], 
                                      num_variations: int = 5) -> List[Tuple[int, int]]:
        """
        Generate variations of tile grid sizes for training.
        
        Args:
            base_tile_grid_size: Base tile grid size (e.g., (8, 8))
            num_variations: Number of variations to generate
            
        Returns:
            List of tile grid size tuples
        """
        tile_grid_sizes = []
        base_size = base_tile_grid_size[0]  # Assuming square grid for simplicity
        
        try:
            for i in range(-num_variations, num_variations + 1):
                # Generate tile grid size variations
                new_size = max(4, base_size + i)  # Ensure a reasonable minimum size
                tile_grid_sizes.append((new_size, new_size))  # Assuming square grid
                
            return tile_grid_sizes
            
        except Exception as e:
            self.logger.error(f"Error generating tile grid variations: {str(e)}")
            raise
