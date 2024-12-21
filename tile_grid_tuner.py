import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Tuple, Dict, Any
import joblib
from data_augmentation import DataAugmentation


class ParameterTuner:
    """
    Machine learning model for optimizing blur kernel parameters and tile grid size.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_fitted = False
        
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract relevant features from image for both kernel size and tile grid size prediction.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        try:
            features = []
            
            # Color features (HSV histogram)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            for channel in range(3):
                hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
                features.extend(hist.flatten())
                
            # Edge features
            edges = cv2.Canny(image, 100, 200)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            features.extend(edge_hist.flatten())
            
            # Texture features (Gabor filters)
            num_filters = 4
            ksize = 21
            for theta in np.arange(0, np.pi, np.pi / num_filters):
                kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0)
                filtered = cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
                
            # Image statistics
            for channel in cv2.split(image):
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.max(channel),
                    np.min(channel)
                ])
                
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
            
    def prepare_dataset(self, images: List[np.ndarray], 
                       tile_grid_sizes: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for training.
        
        Args:
            images: List of input images
            tile_grid_sizes: Corresponding tile grid sizes (as tuple of height, width)
            
        Returns:
            Features and labels for training
        """
        try:
            X = []
            y = []
            
            for image, tile_grid_size in zip(images, tile_grid_sizes):
                features = self.extract_features(image)
                X.append(features)
                y.append(tile_grid_size)  # Append the entire tuple (height, width)
                
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise
            
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model with hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training labels (tile_grid_size as tuple of height, width)
        """
        try:
            X_scaled = self.scaler.fit_transform(X)
            # Define parameter grid for optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [0.5, 'sqrt']
            }
            
            # Initialize base model
            base_model = RandomForestRegressor(random_state=42)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels (tile_grid_size as tuple of height, width)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def predict_tile_grid_size(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Predict optimal tile grid size for CLAHE (clip_limit and tile_grid_size) for new image.
        
        Args:
            image: Input image
            
        Returns:
            Predicted tile grid size as (height, width)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model and scaler must be fitted before making predictions. "
                               "Either train a new model or load a saved model first.")
                
            features = self.extract_features(image)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_size = self.model.predict(features_scaled)[0]
            
            # Ensure minimum size (for example, you can impose a minimum size of 4x4)
            predicted_size = tuple(map(lambda x: max(4, int(x)), predicted_size))
            
            return predicted_size
            
        except Exception as e:
            self.logger.error(f"Error predicting tile grid size: {str(e)}")
            raise
            
    def save_model(self, filepath: str) -> None:
        """
        Save trained model and scaler.
        
        Args:
            filepath: Path to save model
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'best_params': self.best_params
            }
            joblib.dump(model_data, filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: str) -> None:
        """
        Load trained model and scaler.
        
        Args:
            filepath: Path to load model from
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_params = model_data['best_params']
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise


# Main function
def main():
    try:
        # Initialize tuner
        tuner = ParameterTuner()
        
        # Either load a pre-trained model
        try:
            tuner.load_model('mm_tile_grid_model.joblib')
        except FileNotFoundError:
            # Or train a new model if no saved model exists
            print("No saved model found. Training new model...")
            
            # Load and prepare training data
            original_image = cv2.imread('C:/Users/justi/Documents/Western University/Year 3/AISE3350/Project/test.jpg')
            augmented_images = augmenter.augment_image(original_image, num_augmentations=10)
            tile_grid_sizes = augmenter.generate_tile_grid_size_variations((8, 8), num_variations=5)
            
            # Prepare dataset
            X, y = tuner.prepare_dataset(augmented_images, tile_grid_sizes * len(augmented_images))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            tuner.train_model(X_train, y_train)
            tuner.save_model('mm_tile_grid_model.joblib')
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


# Initialize components
augmenter = DataAugmentation(output_dir="C:/Users/justi/Documents/Western University/Year 3/AISE3350/Project/augmented_images")
tuner = ParameterTuner()

# Load and augment training images
original_image = cv2.imread('C:/Users/justi/Documents/Western University/Year 3/AISE3350/Project/test.jpg')
augmented_images = augmenter.augment_image(original_image, num_augmentations=10)

# Generate tile grid size variations
tile_grid_sizes = augmenter.generate_tile_grid_variations((8, 8), num_variations=5)

# Prepare dataset
X, y = tuner.prepare_dataset(augmented_images, tile_grid_sizes * len(augmented_images))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
tuner.train_model(X_train, y_train)

# Evaluate model
metrics = tuner.evaluate_model(X_test, y_test)
print("Model performance:", metrics)

# Save model for future use
tuner.save_model('mm_tile_grid_model.joblib')

# Use model to predict optimal tile grid size for new image
new_image = cv2.imread('C:/Users/justi/Documents/Western University/Year 3/AISE3350/Project/test.jpg')
optimal_tile_grid_size = tuner.predict_tile_grid_size(new_image)
print(f"Optimal tile grid size: {optimal_tile_grid_size}")

if __name__ == "__main__":
    main()