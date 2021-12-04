# projet-CNES

Repository containing our master's project on satellite imaging.

The main goal is to create an "impact map" containing information about wildfires, such as the affected area (crops, forests, etc.) and the impact of the ensuing smoke using the wind's direction.

To-do list:

- [X] get wind directions with dates

- create gradient map? or exploit the data somehow

- [ ] find new API for land cover and wind information
- [ ] integrate new scripts ([make_gif](make_gif.py) and [test_plot](test_plot.py)) into [utils.py](utils) and [earth_engine](earth_engine.ipynb) respectively
- [ ] make the process as automatic as possible
