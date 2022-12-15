import numpy as np
import tensorflow as tf
from dream_utils import Dreamer
from batched_image_param import BatchedAutoImageParam

def render_icons(activations, model, layer, iterations, n_attempts, icon_size, step_size):
    dream_model = Dreamer(model, device = "cpu")
    param = BatchedAutoImageParam(batch = activations.shape[0], height = icon_size, width = icon_size, device = "cuda")
    images = dream_model.render(
    image_parameter = param,
    layers = layer,
    lr = 2e-4,
    # weight_decay= 1e-1,
    iters = 2,  ## very mild
    custom_func = None
    )
    return image_parameter

def grid(xpts=None, ypts=None, grid_size=(4,4), x_extent=(0., 1.), y_extent=(0., 1.)):
    xpx_length = grid_size[0]
    ypx_length = grid_size[1]

    xpt_extent = x_extent
    ypt_extent = y_extent

    xpt_length = xpt_extent[1] - xpt_extent[0]
    ypt_length = ypt_extent[1] - ypt_extent[0]

    xpxs = ((xpts - xpt_extent[0]) / xpt_length) * xpx_length
    ypxs = ((ypts - ypt_extent[0]) / ypt_length) * ypx_length

    ix_s = range(grid_size[0])
    iy_s = range(grid_size[1])
    xs = []
    for xi in ix_s:
        ys = []
        for yi in iy_s:
            xpx_extent = (xi, (xi + 1))
            ypx_extent = (yi, (yi + 1))

            in_bounds_x = np.logical_and(xpx_extent[0] <= xpxs, xpxs <= xpx_extent[1])
            in_bounds_y = np.logical_and(ypx_extent[0] <= ypxs, ypxs <= ypx_extent[1])
            in_bounds = np.logical_and(in_bounds_x, in_bounds_y)

            in_bounds_indices = np.where(in_bounds)[0]
            ys.append(in_bounds_indices)
        xs.append(ys)
    return np.asarray(xs)
    
def render_layout(model, layer, activations, xs, ys, n_steps, n_attempts=2, min_density=0, grid_size=(3, 3), icon_size=3000, x_extent=(0., 1.0), y_extent=(0., 1.0)):
    grid_layout = grid(xpts=xs, ypts=ys, grid_size=grid_size, x_extent=x_extent, y_extent=y_extent)
    icons = []
    X = []
    Y = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            indices = grid_layout[x, y]
            if len(indices) > min_density:
                # average_activation = np.average(activations[indices], axis=0)
                average_activation = activations[indices[0]]
                icons.append(average_activation)
                X.append(x)
                Y.append(y)

    icons = np.asarray(icons)
    print ("Icons:", icons.shape)
    
    icon_batch = render_icons(icons, model, layer, n_steps, n_attempts, icon_size, step_size = 0.01)
    print ("Icon batch:", icon_batch.shape)

    canvas = np.ones((icon_size * grid_size[0], icon_size * grid_size[1], 3))
    for i in range(icon_batch.shape[0]):
        icon = icon_batch[i]
        y = int(X[i])
        x = int(Y[i])
        canvas[(grid_size[0] - x - 1) * icon_size:(grid_size[0] - x) * icon_size, (y) * icon_size:(y + 1) * icon_size] = icon

    return canvas