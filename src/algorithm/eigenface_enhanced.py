import math
import numpy as np
import pyvision.other
import pyvision.other.normalize
from utils.helpers import log_runtime
from sklearn.decomposition import IncrementalPCA
import algorithm.weber as weber
import pyvision
import cv2

DEFAULT_SIGMA = 3.0


class EnhancedEigenface:
    """
    The enhanced Eigenface model for face recognition by Francisco and Seraspi (2024).
    """

    def __init__(self, X, y, M, fst=None, fct=None, avg_ws=True):
        """
        Parameters:
                X (np.ndarray): Training data.
                y (np.ndarray): Training labels.
                M (int): Number of eigenvectors to keep.
                fst (float): Face space threshold.
                fct (float): Face class threshold.
                avg_ws (bool): Whether to use the average weight space.
        """
        self._X = X
        self._y = y
        self._M = M
        self._fst = fst or math.inf
        self._fct = fct or math.inf
        self._avg_ws = avg_ws

    def set_params(self, M=None, fst=None, fct=None, avg_ws=None):
        """
        Set the parameters of the Eigenface model.

        Parameters:
                M (int): Number of eigenvectors to keep.
                fst (float): Face space threshold.
                fct (float): Face class threshold.
                avg_ws (bool): Whether to use the average weight space.

        Returns:
                None

        Raises:
                Exception: If no parameters are set.

        Note:
                Call .fit() first before changing parameters
        """
        if M is None and fst is None and fct is None and avg_ws is None:
            raise Exception("Set at least 1 parameter")

        self._M = M if M is not None else self._M
        self._fst = fst if fst is not None else self._fst
        self._fct = fct if fct is not None else self._fct
        self._avg_ws = avg_ws if avg_ws is not None else self._avg_ws

        try:
            self._calculate_weight_space()
        except:
            raise Exception("Call .fit() first before changing parameters")

    @log_runtime
    def fit(self):
        """
        Fit the Eigenface model to the training data.

        Returns:
                None
        """
        # == Equalize the histogram of the images ==
        self._X_eq_hist_sqi = EnhancedEigenface.eq_hist_sqi_wld(self._X)

        # == Flatten each image into a tall vector of size wh ==
        self._X_flat = EnhancedEigenface.flatten(self._X_eq_hist_sqi)

        # == Calculate the eigenfaces ==
        # Get the mean face of the training set
        self._mean_face = EnhancedEigenface.mean(self._X_flat)
        # Normalize each face image
        self._X_norm = EnhancedEigenface.normalize(self._X_flat, self._mean_face)
        # Perform PCA using SVD
        self._eigenvectors, self._eigenvalues = EnhancedEigenface.PCA(self._X_norm)

        # == Calculate the M-dimensional weight space for each face + the thresholds ==
        self._calculate_weight_space()

    @log_runtime
    def predict(self, X_unknown):
        """
        Predict the labels of the unknown faces.

        Parameters:
                X_unknown (np.ndarray): Unknown faces to predict.

        Returns:
                predicted_labels (np.ndarray): Predicted labels of the unknown faces.
                min_weight_distances (np.ndarray): Minimum weight distances of the unknown faces.
                projection_distances (np.ndarray): Distance between each face and their projection.
        """
        # Equalize the histogram of the images
        X_unknown_eq_hist_sqi = EnhancedEigenface.eq_hist_sqi_wld(X_unknown)
        # Flatten
        X_unknown_flat = EnhancedEigenface.flatten(X_unknown_eq_hist_sqi)
        # Normalize
        X_unknown_norm = EnhancedEigenface.normalize(X_unknown_flat, self._mean_face)
        # Calculate weight space for the unknown face
        X_unknown_weights = EnhancedEigenface.get_weights(
            X_unknown_norm, self._get_eigenfaces()
        )
        X_weights, y = self._get_weight_space()

        weight_distances = EnhancedEigenface.get_weight_distances(
            X_unknown_weights, X_weights
        )
        projection_distances = EnhancedEigenface.get_projection_distances(
            X_unknown_flat, X_unknown_weights, self._get_eigenfaces(), self._mean_face
        )

        min_weight_distances = np.min(weight_distances, axis=0)
        predicted_indices = np.argmin(weight_distances, axis=0)
        temp_labels = y[predicted_indices]

        predicted_labels = []

        for i in range(len(min_weight_distances)):
            min_weight_distance = min_weight_distances[i]
            projection_distance = projection_distances[i]

            if projection_distance > self._fst:
                label = -1
            elif projection_distance < self._fst and min_weight_distance > self._fct:
                label = -2
            else:
                label = temp_labels[i]

            predicted_labels.append(label)

        predicted_labels = np.array(predicted_labels)

        return predicted_labels, min_weight_distances, projection_distances

    def report(self, X_test, y_test):
        """
        Run the Eigenface model on the test data and return the accuracy, precision, recall, and number of rejected faces.

        Parameters:
                X_test (np.ndarray): Test data.
                y_test (np.ndarray): Test labels.

        Returns:
                (accuracy, precision, recall, rejected) (tuple): Evaluation metrics.
                (predicted_labels, min_weight_distances, projected_distances) (tuple): Predicted labels, minimum weight distances, and projected distances.
        """

        predicted_labels, min_weight_distances, projection_distances = self.predict(
            X_test
        )

        # Calculate metrics
        # True positives: Correctly identified faces
        true_positives = np.sum((y_test == predicted_labels) & (y_test > -1))
        # False positives: Unknown faces not rejected
        false_positives = np.sum(
            (predicted_labels > -1) & (y_test < 0) | (y_test == predicted_labels)
        )
        # False negatives: Known faces not correctly identified
        false_negatives = np.sum((predicted_labels != y_test) & (y_test > -1))
        # True negatives: Unknown faces rejected
        true_unknowns = np.sum((predicted_labels < 0) & (y_test < 0))

        accuracy = (true_positives + true_unknowns) / len(y_test)

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0

        rejected = np.sum(predicted_labels < 0)

        return (accuracy, precision, recall, rejected), (
            predicted_labels,
            min_weight_distances,
            projection_distances,
        )

    def score(self, X_test, y_test):
        """
        Calculate the accuracy of the Eigenface model on the test data.

        Parameters:
                X_test (np.ndarray): Test data.
                y_test (np.ndarray): Test labels.

        Returns:
                float: Accuracy of the model.
        """
        predicted_labels, _, _ = self.predict(X_test)
        accuracy = np.sum(y_test == predicted_labels) / len(y_test)

        return accuracy

    def _get_weight_space(self):
        """
        Get the weight space and labels.

        Returns:
                (np.ndarray, np.ndarray): Weight space and labels.
        """
        if self._avg_ws:
            return self._avg_X_weights, self._y_avg
        return self._X_weights, self._y

    def _get_eigenfaces(self):
        """
        Get the eigenvectors (eigenfaces) of the dataset.

        Returns:
                np.ndarray: the M eigenvectors (eigenfaces) to keep where M << N
        """
        # Get the M eigenvectors (eigenfaces) to keep where M << N
        return self._eigenvectors[: self._M]

    def _calculate_weight_space(self):
        """
        Calculate the weight space for the dataset.

        Returns:
                None
        """
        self._X_weights = EnhancedEigenface.get_weights(
            self._X_norm, self._get_eigenfaces()
        )
        # Re-calculate weight space by averaging per label
        self._avg_X_weights, self._y_avg = EnhancedEigenface.get_averaged_weight_space(
            self._X_weights, self._y
        )

    @staticmethod
    def eq_hist_sqi_wld(X, sigma=DEFAULT_SIGMA):
        """
        Equalize the histogram of the images and calculate the Self-Quotient Image (SQI).
        """
        hist_sqi = EnhancedEigenface.sqi(X, sigma)
        # hist_sqi = EnhancedEigenface.equalize_hist(X)
        hist_sqi = EnhancedEigenface.wld(hist_sqi)

        return hist_sqi

    @staticmethod
    def equalize_hist(X):
        """
        Equalize the histogram of the images.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: Equalized dataset.
        """
        eq_hist = np.array([cv2.equalizeHist(x) for x in X])

        return eq_hist

    @staticmethod
    def wld(X):
        """
        Calculate the Weber Local Descriptor (WLD) of the images.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: WLD of the dataset.
        """
        wld = np.array([weber.get_weber_descriptor(x) for x in X])
        wld = 255 - wld
        wld = (wld - np.min(wld)) / (np.max(wld) - np.min(wld)) * 255
        wld = wld.astype(np.uint8)

        return wld

    @staticmethod
    def sqi(X, sigma=DEFAULT_SIGMA):
        """
        Calculate the Self-Quotient Image (SQI) of the images.
        """
        sqi = np.array(
            [pyvision.other.normalize.selfQuotientImage(x, sigma) for x in X]
        )
        # sqi = np.clip(sqi, 0, 1)
        sqi = (sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi)) * 255
        sqi = sqi.astype(np.uint8)

        return sqi

    @staticmethod
    def flatten(X):
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

    @staticmethod
    def mean(X):
        """
        Calculate the mean face of the dataset.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: Mean face of the dataset.
        """
        return np.mean(X, axis=0)

    @staticmethod
    def normalize(X, mean_face):
        """
        Normalize the images by subtracting the mean face.

        Parameters:
                X (np.ndarray): Dataset of images.
                mean_face (np.ndarray): Mean face of the dataset.

        Returns:
                np.ndarray: Normalized dataset.
        """
        return X - mean_face

    @staticmethod
    def PCA(X):
        """
        Perform Principal Component Analysis (PCA) on the dataset using IncrementalPCA.

        Parameters:
                X (np.ndarray): Dataset of images.

        Returns:
                eigenvectors (np.ndarray): Eigenvectors of the dataset.
                eigenvalues (np.ndarray): Eigenvalues of the dataset.
        """
        ipca = IncrementalPCA()
        eigenvectors = ipca.fit(X).components_
        eigenvalues = ipca.explained_variance_

        return eigenvectors, eigenvalues

    @staticmethod
    def get_weights(X, eigenfaces):
        """
        Get the M-dimensional weight space for each face.

        Parameters:
                X (np.ndarray): Dataset of images.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.

        Returns:
                np.ndarray: Weight space of the dataset.
        """
        if len(X.shape) < 2:
            return np.dot(X, eigenfaces.T)
        else:
            return np.array([np.dot(x, eigenfaces.T) for x in X])

    @staticmethod
    def get_averaged_weight_space(weights, y):
        """
        Get the averaged weight space for each label.

        Parameters:
                weights (np.ndarray): Weight space of the dataset.
                y (np.ndarray): Labels of the dataset.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.

        Returns:
                (np.ndarray, np.ndarray): Averaged weight space and labels.
        """
        unique_labels = np.unique(y)
        averaged_weights = []
        averaged_labels = []

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            label_weights = weights[label_indices]
            average_weight = np.mean(label_weights, axis=0)
            averaged_weights.append(average_weight)
            averaged_labels.append(label)

        return np.array(averaged_weights), np.array(averaged_labels)

    @staticmethod
    def reconstruct(X_weights, eigenfaces, mean_face):
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

    @staticmethod
    def get_weight_distances(W_unknown, X_weights):
        """
        Calculate the distances between the weight spaces using the Manhattan distance.

        Parameters:
                W_unknown (np.ndarray): Unknown weight space.
                X_weights (np.ndarray): Known weight space.

        Returns:
                np.ndarray: Distances between the weight spaces.
        """
        return np.sum(np.abs(X_weights[:, None] - W_unknown[None, :]), axis=2)

    @staticmethod
    def get_projection_distances(X, W, eigenfaces, mean_face):
        """
        Calculate the distances between the face and its projection using the Manhattan distance.

        Parameters:
                X (np.ndarray): Dataset of images.
                W (np.ndarray): Weight space of the dataset.
                eigenfaces (np.ndarray): Eigenvectors of the dataset.
                mean_face (np.ndarray): Mean face of the dataset.

        Returns:
                np.ndarray: Distances between the face and its projection.
        """
        projections = EnhancedEigenface.reconstruct(W, eigenfaces, mean_face)
        diff = np.abs(X - projections)
        diff_sum = np.sum(diff, axis=1)
        return diff_sum

    @staticmethod
    def get_face_distances(X_0, X_1):
        """
        Calculate the distances between the face images using the Manhattan distance.

        Parameters:
                X_0 (np.ndarray): Dataset of images.
                X_1 (np.ndarray): Dataset of images.

        Returns:
                np.ndarray: Distances between the face images.
        """
        diff = np.abs(X_0 - X_1)
        diff_sum = np.sum(diff, axis=1)
        return diff_sum