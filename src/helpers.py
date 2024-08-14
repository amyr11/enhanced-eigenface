import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

def log_runtime(func):
	"""
	Decorator to log the runtime of a function.

	Parameters:
		func (function): Function to log the runtime of.
	"""
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		print(f"{func.__name__} took {end_time - start_time} seconds to run.")
		return result
	return wrapper

def load_images_labels(directory, w, h):
	"""
	Load images and labels from a directory.

	Parameters:
		directory (str): Directory containing the images.
		w (int): Width of the images.
		h (int): Height of the images.
	"""
	images = []
	labels = []

	for filename in os.listdir(directory):
		if filename.endswith('.jpg') or filename.endswith('.png'):
			label, _ = os.path.splitext(filename)[0].split("_")
			label = int(label)
			image_path = os.path.join(directory, filename)
			image = cv2.imread(image_path)
			image = preprocess_image(image, w, h)
			images.append(image)
			labels.append(label)

	data = np.array(images)
	labels = np.array(labels)

	return data, labels

def preprocess_image(image, w, h):
	"""
	Preprocess an image.

	Parameters:
		image (np.ndarray): Image to preprocess.
		w (int): Width of the image.
		h (int): Height of the image.
	"""
	image = cv2.resize(image, (w, h))  # Resize to match Olivetti dataset size
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
	image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
	image = image / 255.0
	return image

def find_face_by_label(X, y, label):
	"""
	Find the first face image with the given label.

	Parameters:
		X (np.ndarray): Dataset of images.
		y (np.ndarray): Labels of the dataset.
		label (str): Label to search for.

	Returns:
		np.ndarray: Face image with the given label.
	"""
	indices = np.where(y == label)[0]
	if len(indices) > 0:
		return X[indices[0]]
	else:
		# Return a black image
		return np.zeros(X.shape[1:], dtype=np.uint8)

def plot_portraits(images, titles, w, h, r=None, c=None, title=None, xsize=None, ysize=None):
	"""
	Plot a grid of images.

	Parameters:
		images (list): List of images to plot.
		titles (list): List of titles for each image.
		w (int): Width of the images.
		h (int): Height of the images.
		r (int): Number of rows in the grid.
		c (int): Number of columns in the grid.
		title (str): Title of the grid.
		xsize (int): Width of the grid.
		ysize (int): Height of the grid.
	
	"""
	total_images = len(images)
	cols = c or int(np.sqrt(total_images))  # Calculate the number of columns
	rows = r or (total_images // cols) + 1  # Calculate the number of rows
	plt.figure(figsize=(xsize or (2.2 * cols), ysize or (2.2 * rows)))
	if title:
		plt.suptitle(title, y=1.02)
	for i in range(len(images)):
		plt.subplot(rows, cols, i + 1)
		plt.imshow(images[i].reshape((w, h)), cmap=plt.cm.gray)
		plt.title(titles[i])
		plt.xticks(())
		plt.yticks(())

def plot_comparison(test_face, test_label, predicted_face, predicted_label):
	"""
	Plot the unknown face and its closest match.

	Parameters:
		test_face (np.ndarray): Unknown face.
		test_label (str): Label of the unknown face.
		predicted_face (np.ndarray): Closest match.
		predicted_label (str): Label of the closest match.

	Returns:
		None
	"""
	plt.figure(figsize=(6, 3))
	plt.subplot(1, 2, 1)
	plt.imshow(test_face, cmap='gray')
	plt.title(f'Unknown Face - {test_label}')
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.imshow(predicted_face, cmap='gray')
	plt.title(f'Closest Match - {predicted_label}')
	plt.axis('off')

	plt.show()

def plot_report(model, X_test, y_test, X_train, y_train, wrong_only=False):
	"""
	Show the unknown faces and their closest matches and print the evaluation metrics.

	Parameters:
		model: Eigenface model to evaluate.
		X_test (np.ndarray): Test dataset.
		y_test (np.ndarray): Labels of the test dataset.
		X_train (np.ndarray): Train dataset.
		y_train (np.ndarray): Labels of the train dataset.
		wrong_only (bool): Whether to show only the wrong matches.

	Returns:
		None
	"""
	# predicted_labels, min_weight_distances, projected_distances = model.predict(X_test)

	(accuracy, precision, recall, rejected), (predicted_labels, min_weight_distances, projection_distances) = model.report(X_test, y_test)

	for i in range(len(X_test)):
		test_label = y_test[i]
		test_face = X_test[i]
		predicted_label = predicted_labels[i]
		predicted_face = find_face_by_label(X_train, y_train, predicted_label)


		if test_label == predicted_label:
			if not wrong_only:
				plot_comparison(test_face, test_label, predicted_face, predicted_label)
				print(f'Correct (wd): {min_weight_distances[i]}')
				print(f'Correct (fd): {projection_distances[i]}')
		else:
			plot_comparison(test_face, test_label, predicted_face, predicted_label)
			print(f'Wrong (wd): {min_weight_distances[i]}')
			print(f'Wrong (fd): {projection_distances[i]}')

	print('\n---')
	print(f'Accuracy: {accuracy}')
	# print(f'Precision: {precision}')
	# print(f'Recall: {recall}')
	# print(f'Rejected: {rejected}')