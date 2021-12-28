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

## Obtaining the code

### Downloading the code

In order to execute the code in this repository, you need to install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads).

Next, clone this project by typing the following commands in a terminal:

```console
> git clone https://github.com/jayalabanda/projet-CNES.git
> cd projet-CNES
```

Alternatively, you can download the repository as a ZIP file by clicking [here](https://github.com/jayalabanda/projet-CNES/archive/refs/heads/main.zip), then unzipping the files into the directory of your choice.

Then, run these commands to create a new environment called "cnes":

```console
> conda env create -f environment.yml
> conda activate cnes
```

_Note: if you want to install the packages manually, it is **highly** recommended to install them using Anaconda's `conda` command and `conda-forge` channel instead of pip, since using both package managers can produce many errors._

### Executing the code

There are two ways of executing the code. You can type:

```console
> python main.py
```

in a console, which will launch the application on the command line, or open the notebook [main.ipynb](main.ipynb) in Jupyter Notebook by executing the command:

```console
> jupyter notebook
```

and navigating to the location of the file in the tab that opens in your browser.

---

## Setup

### Wildfire information

Some previous information is needed to execute the code in the notebook or scripts. This information includes:

* the date of the wildfire in format `YYYY-MM-DD`,
* a pair of coordinates _latitude_ and _longitude_ inside the wildfire, such as $44.5, 4.0$ (more information in [INSTRUCTIONS.md](INTRUCTIONS.md)),
* the true area that burned in hectares. This value is easily retrievable in the news.

This information must be saved in a JSON file saved as `info_NAME.json` inside `data/info_fires/`:

```JSON
{
    "wildifire_date": "YYYY-MM-DD",
    "longitude": XX.XX,
    "latitude": YY.YY,
    "true_area": ZZ.ZZ
}
```

where the latitude, longitude, and true area are floats (without the ""). An example is:

```JSON
{
    "name": "Var",
    "wildfire_date": "2021-08-17",
    "latitude": 43.314677,
    "longitude": 6.451894,
    "true_area": 7100.0
}
```

_Note: the name field is not necessary but is given for clarification, and the date must be in the same format._

A good place to start is the [Prométhée list of wildfires](https://www.promethee.com/default/incendies) website, which gives detailed information about these events, up until the end of 2020 and in the French mediterranean area. If you want global data for 2021, you can also visit the [European Forest Fire Information System](https://effis.jrc.ec.europa.eu/) (EFFIS) website and go to "Current Situation Viewer".

### Sentinel API

Moreover, you will need a free [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home) account. More details on the Sentinel API are available [here](https://sentinelsat.readthedocs.io/en/latest/index.html). Then save your credentials in a JSON file inside the `secrets` folder named `sentinel_api_credentials.json` as follows ("" included):

```JSON
{
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD"
}
```

### Google Earth Engine API

In order to use Google's Earth Engine API, you must create a (free) account at [Earth Engine API](https://developers.google.com/earth-engine) and then authenticate once. Your token will be saved so you will not need to do it again anymore.

### Climate Data Store API

You will also need a free CDS account to retrieve wind data. Instructions are given here: [How to use the CDS API](https://cds.climate.copernicus.eu/api-how-to). **Remember** to install the CDS API client using

```console
> conda install -c conda-forge cdsapi
```

instead of using `pip` if it was not installed during the creation of the conda environment.

---

## File structure

The file structure of the repository is, ideally, as follows:

```console
+---data
|   +---coordinates_files
|   +---geojson_files
|   +---info_fires
|   +---nc_files
+---output
|   +---maps
+---scripts
+---secrets
+---utils
```

where:

* the downloaded images and land cover dataframes are in [data](data/),
* various Python scripts are inside the folder of the same name [scripts](scripts/),
* the utility functions are located inside [utils](utils/),
* wind data files stored as NetCDF (.nc) are located in the `nc_files` folder,
* and your access credentials for the Sentinel API are inside `secrets`

### Downloaded images

The first step is obtaining the images. The downloaded folders can vary in size (up to 1.2 GB each), and the produced TIFF files are also around 1 GB each, so please keep this in mind if you have low storage left in your drive. Although, if you are only interested in using the NDVI, you may delete the folders `R20m` and `R60m` containing the JP2 image files.

Moreover, as the downloaded data are large, the first step of the process may take a long time, depending on your download speed and the server's upload speed.

### Coordinates files

This folder will contain any CSV files created during the execution of the program. These files will contain pairs of latitude and longitude coordinates of the affected areas.

### GeoJSON files

The GeoJSON file format is a special type of file storing geospatial data, in a similar format to JSON files. More in-depth information is available at [ArcGIS Online](https://doc.arcgis.com/en/arcgis-online/reference/geojson.htm).

A step-by-step guide to creating a GeoJSON file for an area of interest is given inside the [instructions guide](INSTRUCTIONS.md).

---

Contact details

If you have any questions or encounter any problems, please contact me at my email: [ayalaban@insa-toulouse.fr](mailto:ayalaban@insa-toulouse.fr), or open a pull request.

---

To-do list:

* [ ] create convex hull figures from fire coordinates
* [ ] continue working on [main.py](main.py)
* [ ] clean and test new scripts inside [utils](utils/) folder
* [ ] make the process as automatic as possible
