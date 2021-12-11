import json
import numpy as np
import matplotlib.pyplot as plt
# pylint: disable=no-name-in-module
from scipy.spatial import ConvexHull

with open('data/weather_api.json') as data_file:
    data = json.load(data_file)

theta = []
r = []
for day in data['days']:
    for hour in day['hours']:
        theta.append(hour['winddir'])
        r.append(hour['windspeed'])

theta = np.radians(theta)
x = r * np.cos(theta)
y = r * np.sin(theta)
hull = ConvexHull(np.array([x, y]).T)

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'o', ms=1)
for simplex in hull.simplices:
    plt.plot(x[simplex], y[simplex], 'k-')
for i in range(len(theta)):
    plt.arrow(0, 0, x[i], y[i], color='k', alpha=0.5, width=0.1)
plt.show()
