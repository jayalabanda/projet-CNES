import itertools

import matplotlib.pyplot as plt
import numpy as np


def split_image(image, fragment_count):
    """Split images into fragments.
       Allows to select the potion(s) of the image to be used.

    Args:
        image (image): image to be split
        fragment_count (int): number of fragments to be created

    Returns:
        split_image (array): array of the split image
    """
    n = range(fragment_count)
    frag_size = int(image.shape[0] / fragment_count)
    split_image = {}

    for y, x in itertools.product(n, n):
        split_image[(x, y)] = image[y * frag_size: (y + 1) * frag_size,
                                    x * frag_size: (x + 1) * frag_size]
    return split_image


def plot_split_image(split_image, fragment_count):
    """Plots all of the fragmented images.
       Fragmented images come from the split_image() function.

    Args:
        split_image (array): array of the split image. See split_image()
        fragment_count (int): number of fragments

    Returns:
        None
    """
    n = range(fragment_count)
    _, axs = plt.subplots(fragment_count, fragment_count, figsize=(10, 10))
    for y, x in itertools.product(n, n):
        axs[y, x].imshow(split_image[(x, y)])
        axs[y, x].axis('off')
    plt.tight_layout()
    plt.show()


def threshold_filter(image, threshold):
    """Puts all values below threshold to 0.

    Args:
        image: already imported image
        threshold (float): threshold value

    Returns:
        image: image where all values below threshold are set to 0
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
    """
    Takes 4 images of SAME SIZE, merges them together to get a lager field of view
    along with a bigger final picture.

    Args:
        image_array (list): list of the 4 images that need to be merged.
            First image: upper left. Second image: upper right.
            Third image: lower left. Fourth image: lower right.

    Returns:
        final_mage: one final image that has all 4 images merged together
    """
    # image1 = image_array[0]
    # image2 = image_array[1]
    # image3 = image_array[2]
    # image4 = image_array[3]
    # # get the shapes of the initial images to make an image that is twice as big
    # n, m = image1.shape
    # final_image = np.zeros((2 * n, 2 * m), np.float64)
    # final_image[:n, :m] = image1
    # final_image[n:, :m] = image2
    # final_image[:n, m:] = image3
    # final_image[n:, m:] = image4

    final_image = np.array(
        [np.hstack(image_array[:2]), np.hstack(image_array[2:])],
        dtype=np.float64)
    final_image = np.vstack(final_image)
    return final_image


def merge_images(n_images, images, horizontal=True):
    """Merge images together.

    Args:
        n_images (int): number of images to merge. Can equal 2 or 4.
        images (list): list of images to merge.

    Raises:
        ValueError: Raises an error if the number of images does not match the shape of the list of images
        ValueError: Raises an error if the number of images is not 2 or 4.
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


def imshow(img, title, **kwargs):
    """This is a wrapper for matplotlib's imshow function.

    Args:
        img: image to display
        title: title of the image
        **kwargs: additional arguments to pass to matplotlib.pyplot.imshow
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img, **kwargs)
    plt.title(title)
    plt.show()


def plot_comparison(original, filtered, filter_name):
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
