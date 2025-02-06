import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

def psf_blur_add_noise(image: np.ndarray, sigma: float = 1.0, noise_level: float = 0.01) -> np.ndarray:
    """
    Apply PSF-like blurring to the image using a Gaussian filter and add noise.

    Parameters:
    - image: np.ndarray, the input image (assumed to be a 2D grayscale or 3D color array).
    - sigma: float, the standard deviation for Gaussian kernel (blurring strength).
    - noise_level: float, the standard deviation of Gaussian noise to be added.

    Returns:
    - np.ndarray, the blurred and noisy image.
    """
    if image.ndim == 3:  # Color image
        blurred_image = np.zeros_like(image)
        for i in range(image.shape[2]):  # Apply blur to each channel separately
            blurred_image[..., i] = gaussian_filter(image[..., i], sigma=sigma)
    else:  # Grayscale image
        blurred_image = gaussian_filter(image, sigma=sigma)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape)

    # Add noise to the blurred image
    noisy_blurred_image = blurred_image + noise

    # Clip values to maintain valid pixel range [0, 1] for normalized image
    noisy_blurred_image = np.clip(noisy_blurred_image, 0, 1)

    return noisy_blurred_image

data_train = np.load('../data/galaxy_dataset/train/x_train_desi.npy')
data_test = np.load('../data/galaxy_dataset/test/x_test_desi.npy')

for i in range(6):
    sigma=0.5*i
    original_data_train = data_train
    original_data_test = data_test
    for i, img in enumerate(original_data_train):
        original_data_train[i] = psf_blur_add_noise(original_data_train[i], sigma=sigma, noise_level=0.0)
    np.save(f'x_train_desi_psf_blur_{sigma}.npy', original_data_train)
    for i, img in enumerate(data_test):
        original_data_test[i] = psf_blur_add_noise(original_data_train[i], sigma=sigma, noise_level=0.0)
    np.save(f'x_test_desi_psf_blur_{sigma}.npy', original_data_test)
