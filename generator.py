# * Import Initial datas
from data import combinations, initial_probability

# * Import Wavefunction Collapse Model
from WFC2D import *

model = WFC2D((6, 6), combinations, initial_probability)
model.run(export_animation=True)
model.generate_image()
model.generate_animation()

