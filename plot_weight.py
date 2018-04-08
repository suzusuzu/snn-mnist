import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

imgs = []
for i in range(100):
    input_image = Image.open('image' + str(i) + '.png')
    output_image = ImageOps.grayscale(input_image)
    arr = np.asarray(output_image)
    imgs.append(arr)

rows = []
for i in range(10):
    row = np.array(imgs[i*10])
    for c in range(1, 10):
        row = np.c_[row, imgs[i*10+c]]
    rows.append(row)

mat = np.array(rows[0])
for r in range(1, 10):
    mat = np.r_[mat, rows[r]]

plt.title('weights')
plt.imshow(mat, cmap='gray')
plt.show()
