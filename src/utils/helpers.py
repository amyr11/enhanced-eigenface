import numpy as np
import cv2
import os
import time
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


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
        if filename.endswith(".jpg") or filename.endswith(".png"):
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
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # image = image / 255.0
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


def plot_portraits(
    images, titles, w, h, r=None, c=None, title=None, xsize=None, ysize=None
):
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
    if r and not c:
        cols = (total_images + r - 1) // r  # Calculate the number of columns
        rows = r
    elif c and not r:
        rows = (total_images + c - 1) // c  # Calculate the number of rows
        cols = c
    else:
        cols = c or int(np.sqrt(total_images))  # Calculate the number of columns
        rows = r or (total_images + cols - 1) // cols  # Calculate the number of rows

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
    plt.imshow(test_face, cmap="gray")
    plt.title(f"Unknown Face - {test_label}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_face, cmap="gray")
    plt.title(f"Closest Match - {predicted_label}")
    plt.axis("off")

    plt.show()


def plot_umap_2d(title, data, labels):
    """
    Plots a 2D UMAP projection of the data.

    Parameters:
    - title (str): The title of the plot.
    - data (numpy.ndarray): The high-dimensional data to be reduced and plotted.
    - labels (numpy.ndarray): The labels corresponding to the data points.
    """

    # Reduce the dimensionality of the data using UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_data = umap_reducer.fit_transform(data)

    # Create a scatter plot with each label plotted in a different color
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="rocket_r", s=5
    )
    plt.colorbar(scatter, label="Label")
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.show()


def plot_umap_3d_with_filter(title, data, labels):
    """
    Plots a 3D UMAP projection with a dropdown filter to select which labels are visible.

    Parameters:
    - title (str): The title of the plot.
    - data (numpy.ndarray): The high-dimensional data to be reduced and plotted.
    - labels (numpy.ndarray): The labels corresponding to the data points.
    """

    # Get unique labels and the number of classes
    unique_y_labels = np.unique(labels)
    num_classes = len(unique_y_labels)

    # Generate a custom color palette
    if num_classes <= len(px.colors.qualitative.Vivid):
        color_palette = px.colors.qualitative.Vivid
    else:
        color_palette = pc.sample_colorscale(
            "Turbo", [n / num_classes for n in range(num_classes)]
        )

    # Create a color map for the labels
    color_map = {
        label: color_palette[i % len(color_palette)]
        for i, label in enumerate(unique_y_labels)
    }
    point_colors = [color_map[label] for label in labels]

    # Reduce the dimensionality of the data using UMAP
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    reduced_data = umap_reducer.fit_transform(data)

    # Create a 3D scatter plot with each label plotted separately for filtering
    scatter_traces = []
    for label in unique_y_labels:
        indices = np.where(labels == label)
        scatter_traces.append(
            go.Scatter3d(
                x=reduced_data[indices, 0].flatten(),
                y=reduced_data[indices, 1].flatten(),
                z=reduced_data[indices, 2].flatten(),
                mode="markers",
                name=f"Label {label}",
                marker=dict(
                    size=5,
                    color=color_map[label],
                    opacity=0.8,
                ),
                text=[f"Label: {label}"]
                * len(indices[0]),  # Display the label on hover
                hoverinfo="text",
                visible=True,  # Set all traces visible initially
            )
        )

    # Create a button for each label to toggle visibility
    buttons = []
    for i, label in enumerate(unique_y_labels):
        visibility = [False] * len(scatter_traces)  # Hide all traces initially
        visibility[i] = True  # Only show the selected label

        buttons.append(
            dict(
                label=f"Label {label}",
                method="update",
                args=[
                    {"visible": visibility},  # Update visibility of the traces
                    {"title": f"UMAP of Label {label}"},
                ],  # Update the plot title
            )
        )

    # Add an option to show all labels
    buttons.append(
        dict(
            label="Show All",
            method="update",
            args=[
                {"visible": [True] * len(scatter_traces)},  # Show all traces
                {"title": title},
            ],  # Restore the original title
        )
    )

    # Create the layout with the dropdown menu
    fig = go.Figure(data=scatter_traces)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3",
        ),
        width=700,
        height=700,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.17,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )

    # Show the interactive plot
    fig.show()


def plot_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="rocket_r")

    # print classification report
    print(classification_report(y_true, y_pred))

    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
