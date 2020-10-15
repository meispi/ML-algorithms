import numpy as np


K1 = []
K2 = []


def kmeans(c1,c2,x):
    for i in x:
        dist1 = np.sum(np.square(c1-i))
        dist2 = np.sum(np.square(c2-i))
        if dist1 < dist2:
            K1.append(i)
            c1 = ((c1[0]+i[0])/2, (c1[1]+i[1])/2)
        else:
            K2.append(i)
            c2 = ((c2[0]+i[0])/2, (c2[1]+i[1])/2)
    return c1, c2


c1 = (3,5)
c2 = (-3,-5)
x = -2*np.random.rand(100,2)
x1 = 1+2*np.random.rand(50,2)
x[50:100, :] = x1

temp = kmeans(c1,c2,x)

while np.all(np.floor(10000*np.array(temp[0])) != np.floor(10000*np.array(c1))) and np.all(np.floor(10000*np.array(temp[1])) != np.floor(10000*np.array(c2))):
    c1,c2 = temp
    temp = kmeans(c1,c2,x)

print("Co-ordinates of cluster1", c1)
print(K1)
print("Co-ordinates of cluster2", c2)
print(K2)
