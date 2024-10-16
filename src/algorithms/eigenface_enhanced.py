import math
from utils.helpers import log_runtime
from algorithms.others import wld
import numpy as np
from sklearn.decomposition import IncrementalPCA

NOT_A_FACE = -1
UNKNOWN_FACE = -2


class EnhancedEigenface:
    """
    The enhanced Eigenface model for face recognition by Francisco and Seraspi (2024).
    """

    def __init__(self, M, fct=None, fst=None):
        """
        Parameters:
                M (int): Number of eigenvectors to keep.
                fst (float): Face space threshold.
                fct (float): Face class threshold.
                avg_ws (bool): Whether to use the average weight space.
        """
        self._M = M
        self._fst = fst or math.inf
        self._fct = fct or math.inf

    def fit(self, X, y):
        """
        Fit the Eigenface model to the training data.

        Returns:
                None
        """
        self._X = X
        self._y = y

        # Normalize lighting
        self._X_norm_light = self.normalize_lighting(self._X)
        # Flatten each image into a tall vector of size wh
        self._X_flat = self.flatten(self._X_norm_light)
        # Get the mean face of the training set
        self._mean_face = self.get_mean_face(self._X_flat)
        # Normalize each face image
        self._X_norm = self.normalize(self._X_flat, self._mean_face)
        # Get Eigenfaces using PCA
        self._eigenfaces = self.PCA(self._X_norm, self._M)
        # Calculate the M-dimensional weight space and average it for each face classs
        self._X_avg_weights, self._y_avg = self.get_avg_weight_space(
            self._X_norm, self._eigenfaces, self._y
        )

    def predict(self, X_unknown):
        """
        Predict the face classes of the unknown images.

        Parameters:
                X_unknown (np.ndarray): Unknown images.

        Returns:
                np.ndarray: Predicted face classes.
        """
        # Normalize lighting
        X_unknown_norm_light = self.normalize_lighting(X_unknown)
        # Flatten each image into a tall vector of size wh
        X_unknown_flat = self.flatten(X_unknown_norm_light)
        # Normalize each face image
        assert self._mean_face is not None, "Call fit() first"
        X_unknown_norm = self.normalize(X_unknown_flat, self._mean_face)
        # Calculate the M-dimensional weight space for each face classs
        X_unknown_weights = self.get_weight_space(X_unknown_norm, self._eigenfaces)
        # Get the weight distances between the unknown weights and the dataset weights
        weight_distances = self.get_weight_distances(
            X_unknown_weights, self._X_avg_weights
        )
        # Get the projection distances between the face and its projection
        projection_distances = self.get_projection_distances(
            X_unknown_flat, X_unknown_weights, self._eigenfaces, self._mean_face
        )
        # Get the face classes with the nearest weights to the unknown weights
        y_nearest, min_weight_distances = self.get_nearest_face_class(
            weight_distances, self._y_avg
        )
        # Apply thresholds to predictions
        y_pred = self.apply_thresholds(
            y_nearest, min_weight_distances, projection_distances, self._fct, self._fst
        )

        return y_pred

    @classmethod
    def normalize_lighting(cls, X):
        """
        Normalize the lighting of the images.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: Normalized dataset.
        """
        norm_light = wld(X)

        return norm_light

    @classmethod
    def flatten(cls, X):
        """
        Flatten the images into tall vectors of size wh.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: Flattened dataset.
        """
        if len(X.shape) < 3:
            return X.flatten()
        elif len(X.shape) == 3:
            return np.array([x.flatten() for x in X])

    @classmethod
    def get_mean_face(cls, X_flat):
        """
        Calculate the mean face of the dataset.

        Parameters:
                X_flat (np.ndarray): Flattened dataset of images.

        Returns:
                np.ndarray: Mean face of the dataset.
        """
        return np.mean(X_flat, axis=0)

    @classmethod
    def normalize(cls, X_flat, mean_face):
        """
        Normalize the images by subtracting the mean face.

        Parameters:
                X_flat (np.ndarray): Flattened dataset of images.
                mean_face (np.ndarray): Mean face of the dataset.

        Returns:
                np.ndarray: Normalized dataset.
        """
        return X_flat - mean_face

    @classmethod
    def PCA(cls, X_norm, M):
        """
        Perform Principal Component Analysis (PCA) on the dataset using IncrementalPCA.

        Parameters:
                X_norm (np.ndarray): Normalized dataset of images.
                M (int): Number of eigenvectors to keep.

        Returns:
                eigenvectors (np.ndarray): Eigenvectors of the dataset.
                eigenvalues (np.ndarray): Eigenvalues of the dataset.
        """
        ipca = IncrementalPCA()
        eigenvectors = ipca.fit(X_norm).components_

        return eigenvectors[:M]

    def get_weight_space(self, X_norm, eigenfaces):
        """
        Get the M-dimensional weight space for each face.

        Parameters:
                X_norm (np.ndarray): Normalized dataset of images.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.

        Returns:
                np.ndarray: Weight space of the dataset.
        """
        if len(X_norm.shape) < 2:
            return np.dot(X_norm, eigenfaces.T)
        else:
            return np.array([np.dot(x, eigenfaces.T) for x in X_norm])

    def get_avg_weight_space(self, X_norm, eigenfaces, y):
        """
        Get the averaged weight space for each label.

        Parameters:
                X_norm (np.ndarray): Normalized dataset of images.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.
                y (np.ndarray): Labels of the dataset.

        Returns:
                (np.ndarray, np.ndarray): Averaged weight space and labels.
        """
        self._X_weights = self.get_weight_space(X_norm, eigenfaces)

        # Since the weights are averaged, the number of labels per weights should be the same
        unique_labels = np.unique(y)
        averaged_weights = []
        averaged_labels = []

        # Average the weights per label
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            label_weights = self._X_weights[label_indices]
            average_weight = np.mean(label_weights, axis=0)
            averaged_weights.append(average_weight)
            averaged_labels.append(label)

        return np.array(averaged_weights), np.array(averaged_labels)

    def get_weight_distances(self, X_unknown_weights, X_avg_weights):
        """
        Calculate the distances between the weight spaces using the Manhattan distance.

        Parameters:
                X_unknown_weights (np.ndarray): Unknown weight space.
                X_avg_weights (np.ndarray): Dataset weight space.

        Returns:
                np.ndarray: Distances between the weight spaces.
        """
        return np.sum(
            np.abs(X_avg_weights[:, None] - X_unknown_weights[None, :]), axis=2
        )

    def reconstruct(self, X_weights, eigenfaces, mean_face):
        """
        Reconstruct the faces from the weight space.

        Parameters:
                X_weights (np.ndarray): Weight space of the dataset.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.
                mean_face (np.ndarray): Mean face of the dataset.

        Returns:
                np.ndarray: Reconstructed dataset.
        """
        reconstructed = mean_face + np.dot(X_weights, eigenfaces)

        return reconstructed

    def get_projection_distances(self, X_flat, X_weights, eigenfaces, mean_face):
        """
        Calculate the distances between the face and its projection using the Manhattan distance.

        Parameters:
                X_flat (np.ndarray): Flattened array of selected images.
                X_weights (np.ndarray): Weight space of selected images.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.
                mean_face (np.ndarray): Mean face of the dataset

        Returns:
                np.ndarray: Distances between the face and its projection.
        """
        projections = self.reconstruct(X_weights, eigenfaces, mean_face)
        diff = np.abs(X_flat - projections)
        diff_sum = np.sum(diff, axis=1)

        return diff_sum

    def get_nearest_face_class(self, weight_distances, y_avg):
        """
        Get the face classes with the nearest weights to the unknown weights.

        Parameters:
                weight_distances (np.ndarray): Distances of each unknown weight to the dataset weights.
                y_avg (np.ndarray): Averaged labels of the dataset.

        Returns:
                (np.ndarray, np.ndarray): Nearest face classes and distances.
        """
        # Get the face classes with the nearest weights to the unknown weights
        min_weight_distances = np.min(weight_distances, axis=0)
        predicted_indices = np.argmin(weight_distances, axis=0)
        y_nearest = y_avg[predicted_indices]

        return y_nearest, min_weight_distances

    def apply_thresholds(
        self, y_nearest, min_weight_distances, projection_distances, fct, fst
    ):
        """
        Apply thresholds to the predictions.

        Parameters:
                y_nearest (np.ndarray): Nearest face classes.
                min_weight_distances (np.ndarray): Minimum weight distances.
                projection_distances (np.ndarray): Projection distances.
                fct (float): Face class threshold.
                fst (float): Face space threshold.

        Returns:
                np.ndarray: Predicted face classes.
        """
        y_pred = []

        for i in range(len(min_weight_distances)):
            min_weight_distance = min_weight_distances[i]
            projection_distance = projection_distances[i]

            if projection_distance > fst:
                label = NOT_A_FACE
            elif projection_distance < fst and min_weight_distance > fct:
                label = UNKNOWN_FACE
            else:
                label = y_nearest[i]

            y_pred.append(label)

        return np.array(y_pred)

    def score(self, X_test, y_test):
        """
        Calculate the accuracy of the Eigenface model on the test data.

        Parameters:
                X_test (np.ndarray): Test data.
                y_test (np.ndarray): Test labels.

        Returns:
                float: Accuracy of the model.
        """
        predicted_labels = self.predict(X_test)
        accuracy = np.sum(y_test == predicted_labels) / len(y_test)

        return accuracy
