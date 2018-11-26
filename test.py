import numpy as np

var = 1.0

h,w = 5, 10
x = np.arange(w, dtype=float)
y = np.arange(h, dtype=float)[:,np.newaxis]
x0 = w // 2
y0 = h // 2
mat = np.exp(-0.5 * (pow(x-x0, 2) + pow(y-y0, 2)) / var)
normalized_img = np.zeros((h, w))

print(mat.shape)
print(normalized_img.shape)