import numpy as np


def sqi(X, sigma):
    from pyvision.other.normalize import selfQuotientImage

    """
    Calculate the Self-Quotient Image (SQI) of the images.
    """
    sqi = np.array([selfQuotientImage(x, sigma) for x in X])
    sqi = (sqi - np.min(sqi)) / (np.max(sqi) - np.min(sqi)) * 255
    sqi = sqi.astype(np.uint8)

    return sqi


def get_weber_descriptor(grayscale_image):
    from scipy.signal import convolve2d

    """
    Compute the Weber descriptor of a grayscale image.

    Parameters
    ----------
    grayscale_image : numpy.ndarray
        Grayscale image.

    Returns
    -------
    numpy.ndarray
        Weber descriptor of the grayscale image.

    References:
    J. Chen et al.,
    "WLD: A Robust Local Image Descriptor,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 32, no. 9, pp. 1705-1720,
    Sept. 2010.
    """
    grayscale_image = grayscale_image.astype(np.float64)
    grayscale_image[grayscale_image == 0] = np.finfo(float).eps
    neighbours_filter = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    convolved = convolve2d(grayscale_image, neighbours_filter, mode="same")
    weber_descriptor = convolved - 8 * grayscale_image
    weber_descriptor = weber_descriptor / grayscale_image
    weber_descriptor = np.arctan(weber_descriptor)

    return weber_descriptor


def wld(X):
    """
    Calculate the Weber Local Descriptor (WLD) of the images.

    Parameters:
            X (np.ndarray): Dataset of images.

    Returns:
            np.ndarray: WLD of the dataset.
    """
    wld = np.array([get_weber_descriptor(x) for x in X])
    wld = 255 - wld
    wld = (wld - np.min(wld)) / (np.max(wld) - np.min(wld)) * 255
    wld = wld.astype(np.uint8)

    return wld
