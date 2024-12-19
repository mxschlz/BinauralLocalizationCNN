import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

plt.hsv()

def colors_rgb(x, y, x_min=0, x_max=1, y_min=0, y_max=1, order=0):
    x_gradient = (x - x_min) / (x_max - x_min)
    y_gradient = (y - y_min) / (y_max - y_min)
    combined_gradient = 1 - (x_gradient + y_gradient) * .5

    if order == 0:
        return x_gradient, y_gradient, combined_gradient
    elif order == 1:
        return y_gradient, combined_gradient, x_gradient
    elif order == 2:
        return combined_gradient, x_gradient, y_gradient

# Plotting RGB gradients
order_0 = [colors_rgb(x/256, y/256, 0, 1, 0, 1, 0) for x in range(256) for y in range(256)]
img_0 = np.array(order_0).reshape(256, 256, 3)
order_1 = [colors_rgb(x/256, y/256, 0, 1, 0, 1, 1) for x in range(256) for y in range(256)]
img_1 = np.array(order_1).reshape(256, 256, 3)
order_2 = [colors_rgb(x/256, y/256, 0, 1, 0, 1, 2) for x in range(256) for y in range(256)]
img_2 = np.array(order_2).reshape(256, 256, 3)

# Transform to black and white
img_0_bw = np.dot(img_0, [0.2989, 0.5870, 0.1140])
img_1_bw = np.dot(img_1, [0.2989, 0.5870, 0.1140])
img_2_bw = np.dot(img_2, [0.2989, 0.5870, 0.1140])

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax[0][0].imshow(img_0)
ax[0][0].x_label("Green")
ax[0][0].y_label("Red")

ax[0][1].imshow(img_1)
ax[0][1].x_label("Red")
ax[0][1].y_label("Blue")

ax[0][2].imshow(img_2)
ax[0][2].x_label("Blue")
ax[0][2].y_label("Green")

ax[1][0].imshow(img_0_bw, cmap="gray")
ax[1][1].imshow(img_1_bw, cmap="gray")
ax[1][2].imshow(img_2_bw, cmap="gray")









plt.tight_layout()
plt.show()
