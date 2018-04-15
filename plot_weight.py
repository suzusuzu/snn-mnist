import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

width = 10
num_unit = width*width

imgs = []
for i in range(num_unit):
    input_image = Image.open('image' + str(i) + '.png')
    output_image = ImageOps.grayscale(input_image)
    arr = np.asarray(output_image)
    imgs.append(arr)

rows = []
for i in range(width):
    row = np.array(imgs[i*width])
    for c in range(1, width):
        row = np.c_[row, imgs[i*width+c]]
    rows.append(row)

mat = np.array(rows[0])
for r in range(1, width):
    mat = np.r_[mat, rows[r]]

plt.title('weights')
plt.imshow(mat, cmap='gray')
plt.show()
