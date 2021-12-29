# This script creates a figure to retrieve the legend for the
# assciated land cover values in each dataframe.
# For example, the value '10' in the ESA dataframe is associated with
# the color '#006400' in the legend and corresponds to the value 'Trees'

import matplotlib.colors as mplcols
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

filenames = ['../data/Copernicus_CORINE_Land_Cover.csv',
             '../data/Copernicus_Landcover_100m_Proba-V-C3_Global.csv',
             '../data/ESA_WorldCover_10m_v100.csv',
             '../data/MODIS_LandCover_Type1.csv']

legends = ['legend_CORINE', 'legend_CGLS',
           'legend_ESA', 'legend_MODIS']

titles = ['CORINE Land Cover', 'CGLS Land Cover',
          'ESA World Cover', 'MODIS Land Cover']

for j in range(len(filenames)):
    data = pd.read_csv(filenames[j])
    labels = data['Description'].to_list()
    colors = data['Color'].to_list()
    colors = [mplcols.to_rgb(i) for i in colors]

    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(10, 10))
    with sns.axes_style('darkgrid'):
        for i in range(len(labels)):
            # Note that the line to plot is arbitrary and can be changed
            ax.plot(x, np.exp(x+i), label=labels[i], color=colors[i])
        leg = ax.legend(ncol=2, loc='best', bbox_to_anchor=[1.05, 1],
                        borderaxespad=0., mode='Expand', title=titles[j])
        for line in leg.get_lines():
            line.set_linewidth(4.0)
        for text in leg.get_texts():
            text.set_size('large')
        plt.savefig(legends[j] + '.png', dpi=300)
        plt.show()
