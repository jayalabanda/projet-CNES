import itertools

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from utils.land_coverage import get_tci_file_path
from utm import from_latlon


def imshow(img, title=None, **kwargs):
    """This is a wrapper for matplotlib's `imshow` function.

    Args:
        img: image to display
        title: title of the image
        **kwargs: additional arguments passed to `matplotlib.pyplot.imshow`
    """
    plt.imshow(img, **kwargs)
    if title is not None:
        plt.title(title)


def get_fire_pixels(image_folder, latitude, longitude):
    """Using the coordinates of the wildifre,
    return the pixel row and column inside the 'NDVI difference' image.

    Args:
        image_folder (str): path to the folder where the images are stored
        latitude (float): latitude of the fire
        longitude (float): longitude of the fire
    """
    tci_file_path = get_tci_file_path(image_folder)
    transform = rasterio.open(tci_file_path, driver='JP2OpenJPEG').transform
    zone_number = int(tci_file_path.split("/")[-1][1:3])
    utm_x, utm_y = transform[2], transform[5]

    east, north, _, _ = from_latlon(
        latitude, longitude, force_zone_number=zone_number)

    pixel_column = round((east - utm_x) / 10)
    pixel_row = round((north - utm_y) / - 10)

    return pixel_column, pixel_row


def split_image(image, fragment_count):
    """Split an image into a grid of equally-sided fragments.

    The result is a `fragment_count` x `fragment_count` grid of images.

    Args:
        image (image): image to be split
        fragment_count (int): number of fragments to be created

    Returns:
        split (array): array of the split image
    """
    n = range(fragment_count)
    frag_size = int(image.shape[0] / fragment_count)
    split = {}

    for y, x in itertools.product(n, n):
        split[(x, y)] = image[y * frag_size: (y + 1) * frag_size,
                              x * frag_size: (x + 1) * frag_size]
    return split


def plot_split_image(split_image, fragment_count):
    """Plots all of the fragmented images.

    The split image comes from the `split_image` function.

    Args:
        split_image (array): array of the split image. See `split_image`
        fragment_count (int): number of fragments
    """
    n = range(fragment_count)
    _, axs = plt.subplots(fragment_count, fragment_count, figsize=(10, 10))
    for y, x in itertools.product(n, n):
        axs[y, x].imshow(split_image[(x, y)])
        axs[y, x].axis('off')


def threshold_filter(image, threshold):
    """Puts all values below `threshold` to 0.

    Args:
        image: already imported image
        threshold (float): threshold value

    Returns:
        image: image where all values below `threshold` are set to 0
    """
    temp = image.copy()
    temp[temp < threshold] = 0
    return temp


def calculate_area(sub_image, original_image, resolution=10):
    """Calculates the surface, in squared kilometers, of the burnt area.

    Args:
        sub_image: already imported image after thresholding
        original_image: tiff image obtained from the API
        resolution (int): resolution of the image. Defaults to 10
            (10 means 1 pixel = 10m, etc.)

    Returns:
        area: area of the image in squared kilometers
    """
    count = np.count_nonzero(sub_image)
    original_area = original_image.size * resolution**2 / 1_000_000  # km^2
    sub_image_area = sub_image.size / original_image.size * original_area
    return count / sub_image.size * sub_image_area


def merge_two_images(images, horizontal=True):
    """Merges two images. The left-most or upper image is the first image in the list.

    Args:
        images (list): list of two images

    Returns:
        new_image: concatenated image
    """
    return np.hstack(images) if horizontal else np.vstack(images)


def merge_four_images(image_array):
    """Takes four images of SAME SIZE and merges them together in a 2x2 grid.

    Args:
        image_array (list): list of the four images that are to be merged.
            First image: upper left. Second image: upper right.
            Third image: lower left. Fourth image: lower right.

    Returns:
        final_mage: one final image that has all four images merged together
    """
    final_image = np.array(
        [np.hstack(image_array[:2]), np.hstack(image_array[2:])],
        dtype=np.float64
    )
    final_image = np.vstack(final_image)
    return final_image


def merge_images(n_images, images, horizontal=True):
    """Merge 2 or 4 images together.

    If `n_images` is 2, the images are merged horizontally or vertically.

    If `n_images` is 4, the images are merged in a 2x2 grid.

    Args:
        n_images (int): number of images to merge. Can equal 2 or 4.
        images (list): list of images to merge.

    Raises:
        ValueError: Raises an error if the number of images does not match the shape of the list of images
        ValueError: Raises an error if the number of images is not 2 or 4.

    Returns:
        final_image: merged image
    """
    if n_images != len(images):
        raise ValueError(
            "Number of images must be equal to the length of the image array.")

    if n_images == 2:
        return merge_two_images(images, horizontal)
    elif n_images == 4:
        return merge_four_images(images)
    else:
        raise ValueError("Number of images must be 2 or 4.")


def plot_comparison(original, filtered, filter_name):
    """Plots the original and filtered images side by side.

    This function is used when comparing skimage's `morphology` methods.

    Args:
        original (image): original image
        filtered (image): filtered image
        filter_name (str): name of the filter
    """
    _, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(12, 8), sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
