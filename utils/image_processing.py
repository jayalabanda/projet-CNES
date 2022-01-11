import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utm import from_latlon

from utils.land_coverage import get_tci_file_path


def imshow(img, figsize, title=None, **kwargs):
    """This is a wrapper for matplotlib's `imshow` function,
    adding a colorbar and a title.

    Args:
        img: image to display
        figsize (tuple): size of the figure
        title (str): title of the image
        **kwargs: additional arguments passed to `matplotlib.pyplot.imshow`
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(img, **kwargs)
    ax.set_title(title, {'fontsize': 16})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    return ax


def plot_downloaded_images(fire_name, output_folder, cmap=None, save=False):
    """Plot the created TIFF images from before and after the fire.

    Args:
        fire_name (str): name of the fire
        output_folder (str): path to the folder where the images are stored
        cmap (str): color map to use for the images. Default is `None`
    """
    b = rasterio.open(f"{output_folder}before_{fire_name}.tiff",
                      driver='GTiff').read(1)
    a = rasterio.open(f"{output_folder}after_{fire_name}.tiff",
                      driver='GTiff').read(1)

    images = [b, a]
    titles = ['NDVI Before', 'NVDI After']

    _, axs = plt.subplots(1, 2, figsize=(12, 10))
    for i in range(2):
        im = axs[i].imshow(images[i], cmap=cmap)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if i == 0:
            cb = plt.colorbar(im, cax=cax)
            cb.remove()
        else:
            plt.colorbar(im, cax=cax)
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()
    if save:
        output_folder = f'{output_folder}plots/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(f'{output_folder}{fire_name}_images.png', dpi=200)
    plt.show()


def get_ndvi_difference(output_folder, fire_name, save_diff=False):
    """Plots and returns the NDVI difference between the images.

    Args:
        output_folder (str): path to the folder where the images are stored
        fire_name (str): name of the fire
        save_diff (bool): whether to save the image. Default is `False`

    Returns:
        difference: NDVI difference between the images
    """
    b = rasterio.open(f'{output_folder}before_{fire_name}.tiff').read(1)
    a = rasterio.open(f'{output_folder}after_{fire_name}.tiff').read(1)
    difference = b - a

    if save_diff:
        # save the difference image
        output_file = f'{output_folder}ndvi_difference_{fire_name}.tiff'
        if not os.path.exists(output_file):
            with rasterio.open(fp=output_file,
                               mode='w', driver='GTiff',
                               width=b.shape[1],
                               height=b.shape[0],
                               count=1,
                               crs=b.crs,
                               transform=b.transform,
                               dtype='float64') as diff_img:
                diff_img.write(difference, 1)
        else:
            print('Output file already exists.')

    return difference


def get_fire_pixels(image_folder, latitude, longitude):
    """Using the coordinates of the wildifre,
    return the pixel row and column inside the 'NDVI difference' image.

    Args:
        image_folder (str): path to the folder where the images are stored
        latitude (float): latitude of the fire
        longitude (float): longitude of the fire

    Returns:
        pixel_row (int): row of the fire pixel
        pixel_column (int): column of the fire pixel
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


def plot_location(ax, pixel_column, pixel_row):
    """Plot a red dot on the image at the given pixel location."""
    ax.plot(pixel_column, pixel_row, 'ro',
            markersize=4, label='Wildfire Location')
    ax.legend(fontsize=13, loc='best')


def plot_fire_area(image, v1, v2, h1, h2,
                   pixel_column, pixel_row, **kwargs):
    """Plot the area inside `v1`, `v2`, `h1`, and `h2`.

    Args:
        image (image): image to be delimited
        v1 (int): first vertical line
        v2 (int): second vertical line
        h1 (int): first horizontal line
        h2 (int): second horizontal line
        pixel_column (int): column of the fire pixel
        pixel_row (int): row of the fire pixel
        **kwargs: additional arguments passed to `matplotlib.pyplot.imshow`
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(image, **kwargs)
    ax.plot(pixel_column, pixel_row, 'ro',
            markersize=4, label='Fire Location')
    ax.vlines(v1, ymin=0, ymax=image.shape[0],
              color='r', linestyle='dashed', linewidth=1)
    ax.vlines(v2, ymin=0, ymax=image.shape[0],
              color='r', linestyle='dashed', linewidth=1)
    ax.hlines(h1, xmin=0, xmax=image.shape[1],
              color='r', linestyle='dashed', linewidth=1)
    ax.hlines(h2, xmin=0, xmax=image.shape[1],
              color='r', linestyle='dashed', linewidth=1)
    ax.legend(fontsize=13, loc='best')
    plt.tight_layout()
    plt.show()


def retrieve_fire_area(image, pixel_column, pixel_row,
                       figsize=(10, 10), title='Fire Area', **kwargs):
    """Retrieve the fire area from the image.

    Args:
        image (image): image to be processed
        pixel_column (int): column of the fire pixel
        pixel_row (int): row of the fire pixel
        figsize (tuple): size of the figure. Default is (10, 10)
        title (str): title of the image
        **kwargs: additional arguments passed to `matplotlib.pyplot.imshow`
    """
    n, m = image.shape
    while True:
        try:
            print(f"""Enter the first vertical line.
                Value must be an integer between 0 and {n}:""")
            v1 = int(input())
            assert 0 <= v1 <= n
            print(f"""Enter the second vertical line.
                Value must be an integer between {v1} and {n}:""")
            v2 = int(input())
            assert v1 < v2 <= n
            print(f"""Enter the first horizontal line.
                Value must be an integer between 0 and {m}:""")
            h1 = int(input())
            assert 0 <= h1 <= m
            print(f"""Enter the second horizontal line.
                Value must be an integer between {h1} and {m}:""")
            h2 = int(input())
            assert h1 < h2 <= m
            print(f"""Your inputs:
                1st vertical line: {v1}
                2nd vertical line: {v2}
                1st horizontal line: {h1}
                2nd horizontal line: {h2}
                """)

            fire = image[h1:h2, v1:v2]
            plt.figure(figsize=figsize)
            plt.imshow(fire, **kwargs)
            plt.show()

            sat = input("Are you satisfied with the values? (y/n): ")
            print("\n")
            if sat == "y":
                break
            plot_fire_area(image, v1, v2, h1, h2,
                           pixel_column, pixel_row,
                           **kwargs)
            continue
        except ValueError:
            print("Invalid value. Try again.")
    ax = imshow(fire, figsize, title, **kwargs)
    return fire, h1, v1


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
        threshold (float): threshold value between -1 and 1

    Returns:
        image: image with all values below `threshold` set to 0
    """
    temp = image.copy()
    temp[temp < threshold] = 0
    return temp


def calculate_area(sub_image, original_image, resolution=10):
    """Calculates the surface, in squared kilometers, of the burnt area.

    Args:
        sub_image: already imported image after thresholding
        original_image: tiff image obtained from the API
        resolution (int): resolution of the image. Default is 10
        (10 means 1 pixel = 10m, etc.)

    Returns:
        area: area of the image in squared kilometers
    """
    count = np.count_nonzero(sub_image)
    original_area = original_image.size * resolution**2 / 1_000_000  # km^2
    sub_image_area = sub_image.size / original_image.size * original_area
    return count / sub_image.size * sub_image_area


def get_thresholds_areas(fire, original_image, resolution=10):
    areas = []
    thresholds = np.linspace(0., 1., 200)
    for thr in thresholds:
        tmp = threshold_filter(fire, thr)
        area = round(calculate_area(tmp, original_image, resolution) * 100, 4)
        areas.append(area)

    return thresholds, areas


def get_threshold(thresholds, areas, true_area):
    """Finds the threshold that gives the best approximation of the true area.

    Args:
        thresholds (array): array of the thresholds
        areas (array): array of the areas corresponding to the thresholds
        true_area (float): true area in squared kilometers

    Returns:
        threshold (float): threshold that gives the best approximation of the
        true area
    """
    areas = np.asarray(areas)
    diff = abs(true_area - areas)
    return round(thresholds[np.argmin(diff)], 3)


def plot_area_vs_threshold(thresholds, areas, true_area):
    plt.figure(figsize=(8, 6))
    with sns.axes_style('darkgrid'):
        plt.plot(thresholds, areas)
        plt.hlines(true_area, thresholds[0], thresholds[-1],
                   colors='r', linestyles='dashed')
        plt.xlabel('Threshold')
        plt.ylabel('Burnt Area (ha)')
        plt.title('Calculated Area vs. Threshold')
        plt.legend(['Calculated Area', 'True Value'])


def merge_two_images(images, horizontal=True):
    """Merges two images.
    The left-most or upper image is the first image in the list.

    Args:
        images (list): list of two images
        horizontal (bool): if True, the images are merged horizontally.

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

    If `n_images` is 2, the images are merged horizontally or vertically
    depending on the value of `horizontal`.

    If `n_images` is 4, the images are merged in a 2x2 grid.

    Args:
        n_images (int): number of images to merge. Can equal 2 or 4.
        images (list): list of images to merge.
        horizontal (bool): if True and `n_images` is 2, the images are
        merged horizontally.

    Raises:
        ValueError: Raises an error if the number of images does not match
            the shape of the list of images
        ValueError: Raises an error if the number of images is not 2 or 4.

    Returns:
        final_image: merged image
    """
    if n_images != len(images):
        raise ValueError(
            "Number of images must equal the length of the image array.")

    if n_images == 2:
        return merge_two_images(images, horizontal)
    elif n_images == 4:
        return merge_four_images(images)
    else:
        raise ValueError("Number of images must be 2 or 4.")


def plot_comparison(original, filtered, filter_name, **kwargs):
    """Plots the original and filtered images side by side.
    This function is used when comparing skimage's `morphology` methods.

    Args:
        original (image): original image
        filtered (image): filtered image
        filter_name (str): name of the filter
        **kwargs: keyword arguments passed to `matplotlib.pyplot.imshow`
    """
    _, axs = plt.subplots(
        ncols=2, figsize=(12, 8), sharex=True, sharey=True)
    axs[0].imshow(original, **kwargs)
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(filtered, **kwargs)
    axs[1].set_title(filter_name)
    axs[1].axis('off')
    plt.tight_layout()
