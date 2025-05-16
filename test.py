import geomy
import numpy as np

points = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [1., 1., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 0., 1.],
    [1., 1., 1.],
    [0., 1., 1.]
])


a = geomy.get_volume_hexa(points)
print(a)