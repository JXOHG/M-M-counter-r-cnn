import numpy as np
from sklearn.model_selection import ParameterGrid
import cv2

class ParameterOptimizer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Failed to load image")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.target_mm_count = 37  # Set the target number of M&Ms

    def preprocess_image(self, tile_grid_size, blur_kernel_size, blur_sigma):
        # Convert to LAB color space
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel back with A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Adjust brightness and contrast
        adjusted = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
        
        # Convert to grayscale
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, blur_kernel_size, blur_sigma)
        
        return blurred

    def detect_circles(self, processed_image, min_radius, max_radius, min_dist, param1, param2):
        circles = cv2.HoughCircles(
            processed_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return np.array([])

    def calculate_confidence(self, circles):
        if len(circles) == 0:
            return 0.0
        
        # Calculate confidence based on circle properties
        radii = circles[:, 2]
        radius_variation = np.std(radii) / np.mean(radii)
        detection_confidence = max(0, 1 - radius_variation)
        
        # Calculate confidence based on the number of detected M&Ms
        num_circles = len(circles)
        count_difference = abs(num_circles - self.target_mm_count)
        count_confidence = max(0, 1 - count_difference / self.target_mm_count)
        
        # Combine detection confidence and count confidence
        overall_confidence = (detection_confidence + count_confidence) / 2
        
        return overall_confidence

    def optimize_parameters(self):
        param_grid = {
            'tile_grid_size': [(2,2), (3,3), (4,4), (8,8), (12,12), (16,16)],
            'blur_kernel_size': [(3,3), (5,5), (7,7), (9,9)],
            'blur_sigma': [0.1, 0.25, 0.5, 0.75, 1.0],
            'min_radius': [0, 10, 15, 20, 30, 40],
            'max_radius': [30, 40, 50, 60],
            'min_dist': [30],  # Using a single value as default
            'param1': [50],    # Using a single value as default
            'param2': [30],    # Using a single value as default
        }

        best_confidence = 0
        best_params = None

        for params in ParameterGrid(param_grid):
            if params['min_radius'] >= params['max_radius']:
                continue  # Skip invalid combinations

            processed_image = self.preprocess_image(
                params['tile_grid_size'], 
                params['blur_kernel_size'], 
                params['blur_sigma']
            )
            circles = self.detect_circles(
                processed_image, 
                params['min_radius'], 
                params['max_radius'],
                params['min_dist'],
                params['param1'],
                params['param2']
            )
            confidence = self.calculate_confidence(circles)

            if confidence > best_confidence:
                best_confidence = confidence
                best_params = params

            print(f"Parameters: {params}, Confidence: {confidence:.4f}, Detected M&Ms: {len(circles)}")

        print(f"\nBest parameters: {best_params}")
        print(f"Best confidence: {best_confidence:.4f}")

        return best_params, best_confidence

# Example usage
optimizer = ParameterOptimizer("C:/Users/justi/Documents/Western University/Year 3/AISE3350/Project/test.jpg")
best_params, best_confidence = optimizer.optimize_parameters()

