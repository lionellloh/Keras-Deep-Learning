import numpy as np

# Broadcasting basically means that a smaller tensor will match the shape of the larger tensor
# Axes are added to the smaller tensor to match the ndim of the larger tensor
# The smaller tensor is then repeated along these new axes tro match the full shape of the larger tensor.

x = np.random.random((64,3,32,10))
# print("x", x)
print(x.shape)
y = np.random.random((32,10))
# print("y", y)
print(y.shape)
z = np.maximum(x, y)
print(z.shape)

# Tensor Dot
# Just a typical dot product
a = np.random.random((32,10))
b = np.random.random((10,1))

z = np.dot(a,b)

print(z)

print(z.shape)

# Tensor reshaping
"""For tensor reshaping the number of coefficients must stay the same before and after reshaping.
For example, if x.shape = (20,30) then the product of its new shape after reshaping must 
always = 20 * 30 = 600. """


