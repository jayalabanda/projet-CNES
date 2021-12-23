# CNES Research and Innovation Project

The main goal is to create an "impact map" containing information about wildfires, such as the affected area (crops, forests, etc.) and the impact of the ensuing smoke using the wind's direction.

---

## Introduction

![CNES Logo](https://cnes.fr/sites/default/files/drupal/201708/image/is_logo_2017_logo_charte_carre_bleu_transparent.png)

This project was our M.Eng's Innovation and Research Project at INSA Toulouse, in partnership with the French space agency CNES (National Centre for Space Studies, in French).

This project consisted in detecting, quantifying, and studying the impact of wildfires. To do this, we proceeeded in three main steps:

1. First, we retrieved images from the affected area using Sentinel's mission's images and with using its Python API (details [here](https://github.com/sentinelsat/sentinelsat) and [here](https://sentinelsat.readthedocs.io/en/stable/)).
2. Then, using an index called NDVI¹, which "measures the density of green on an area of land" (Weier and Herring, 2000), we were able to highlight the affected area by measuring the difference in NDVI from before and after the wildfire.
3. The next step consisted in finding out what type of land cover was burned using different datasets, such as Copernicus CORINE Land Cover, or MODIS Land Cover.
4. Finally, by retrieving the wind data from the day of the wildfire, we wanted to analyze the areas that were affected by the ensuing smoke.

The final objective was to automate this process as much as possible, where we only have some information of the wildfire: a pair of coordinates from where it happened, the date, and the true area that burned (our results regarding the computation of the burnt area are shown in the program).

¹[Normalized difference vegetation index](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)

---

## Instructions

### Obtaining the code

In order to execute the code in this repository, you need to install [Anaconda](https://www.anaconda.com/distribution/) and [git](https://git-scm.com/downloads).

Next, clone this project by typing the following commands in a terminal:

```console
> git clone https://github.com/jayalabanda/projet-CNES.git
> cd projet-CNES
```

Alternatively, you can download the repository as a ZIP file by clicking [here](https://github.com/jayalabanda/projet-CNES/archive/refs/heads/main.zip), then unzipping the files into the directory of your choice.

Then, run these commands:

```console
> conda env create -n cnes -f environment.yml
> conda activate cnes
```

_Note: if you want to install the packages manually, it is **highly** recommended to install them using Anaconda's `conda` command and `conda-forge` channel instead of pip, since using both package managers can produce many errors._

There are two ways of executing the code. You can type:

```console
> python main.py
```

in a console, which will ask the user for multiples inputs on the command line, or open the notebook [main.ipynb](main.ipynb) in Jupyter Notebook by executing the command:

```console
> jupyter notebook
```

### File structure

The file structure of the repository is, ideally, as follows:

```console
+---data
|   +---coordinates_files
|   +---geojson_files
|   +---info_fires
|   +---nc_files
+---output
    +---maps
+---scripts
+---secrets
+---utils
```

where:

- the downloaded images and land cover dataframes are in [data](data/),
- Python scripts are inside the folder of the same name
- the utility functions are located inside [utils](utils/)
- and your access credentials for Sentinel API are inside [secrets](secrets/). More details are available inside the [main notebook](main.ipynb).

### Downloaded images

The first step is obtaining the images. The downloaded folder can vary in size (up to 1.2 GB), and the produced TIFF files are also around 1 GB each, so please keep this in mind if you have low storage left in your drive.

Moreover, as the downloaded data are large, the first step of the process may take a long time, depending on your download speed and the server's upload speed.

### Coordinates files

This folder will contain any CSV file created during the execution of the program. These files will contain pairs of latitude and longitude coordinates of the affected areas.

### GeoJSON files

In order to create the GeoJSON files necessary for calling the Sentinel API, please follow the instructions inside [INSTRUCTIONS.md](INSTRUCTIONS.md).

---

#### Contact details

If you have any questions or encounter any problems, please contact me at my email: [ayalaban@insa-toulouse.fr](mailto:ayalaban@insa-toulouse.fr), or open a pull request.

---

To-do list:

- [ ] (optional) find new API for ~~land cover and~~ wind information
- [x] try other land cover data with `earth-engine`
- [ ] create convex hull figures from fire coordinates
- [ ] continue working on [main.py](main.py)
- [ ] clean and test new scripts inside [utils](utils/) folder
- [x] test all functions with a new fire
- [ ] make the process as automatic as possible
- [x] add user input where necessary
- [x] add robustness to scripts
- [x] start documentation
