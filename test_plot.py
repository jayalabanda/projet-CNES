import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
import matplotlib.colors as mplcols

import ee
ee.Initialize()

output_folder = 'output/'

lc = ee.ImageCollection('MODIS/006/MCD12Q1')
lst = ee.ImageCollection('MODIS/006/MOD11A1')
elv = ee.Image('USGS/SRTMGL1_003')

i_date = '2021-08-15'
f_date = '2021-08-17'
lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)

coords_utm = pd.read_csv('data/coords_utm_var.csv')
lc_type = pd.read_csv('data/MODIS_LandCover_Type1.csv')
lc_type['Color'] = lc_type['Color'].apply(lambda x: f'#{x}')

scale = 1000

for choose in range(550, 1001, 50):
    np.random.seed(42)
    covers = []
    random_idxs = np.sort(
        np.random.choice(range(len(coords_utm)),
                         size=choose, replace=False)
    )

    for i in random_idxs:
        u_lat, u_lon = coords_utm.iloc[i]['latitude'], coords_utm.iloc[i]['longitude']
        u_poi = ee.Geometry.Point(u_lon, u_lat)
        time.sleep(0.25)

        try:
            lc_urban_point = lc.first().sample(u_poi, scale).first().get('LC_Type1').getInfo()
            covers.append(lc_urban_point)
        except:
            print('error with earth engine')

    if None in covers:
        covers = [i for i in covers if i]
    covers = dict(Counter(covers))
    print(covers)

    labels = [lc_type.loc[lc_type['Value'] == i]['Description'].values[0]
              for i in covers.keys()]
    labels = [i.split(':')[0] for i in labels]

    colors = [
        lc_type.loc[lc_type['Value'] == i]['Color'].values[0] for i in covers
    ]

    colors = [mplcols.to_rgb(i) for i in colors]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.set_aspect('equal')
    wedges, texts, autotexts = ax.pie(covers.values(),
                                      colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      textprops=dict(color='w'))
    ax.legend(wedges, labels,
              title='Land Cover Type',
              loc='best',
              bbox_to_anchor=(0.9, 0, 0.5, 1),
              prop={'size': 8},
              labelspacing=0.3)
    plt.setp(autotexts, size=6, weight='bold')
    plt.title(f'N={choose}')
    plt.tight_layout()
    plt.savefig(f'{output_folder}pie_{choose}.png')
plt.show()
