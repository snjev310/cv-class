import numpy as np
import cv2
from matplotlib import pyplot as plt

filename = 'calib-object.jpg'

points3d = [[0,0,0,1],
            [0,0,1,1],
            [0,0,2,1],
            [0,0,3,1],
            [0,0,4,1],
            [0,1,0,1],
            [0,2,0,1],
            [0,3,0,1],
            [0,4,0,1],
            [1,0,0,1],
            [2,0,0,1],
            [3,0,0,1],
            [4,0,0,1]]

coordinates = list()

#print([i for i in dir(cv2) if 'EVENT' in i])
cv2.namedWindow('image')
img = cv2.imread(filename, cv2.IMREAD_COLOR)

def clickEventHandler(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print('new cood ({},{}) recorded'.format(x,y))
        coordinates.append([ x, y, 1])

cv2.setMouseCallback('image', clickEventHandler)


while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

A = list()
print(len(coordinates), ' points recorede')
print('x\ty\tX\tY\tZ')
for (x, y, c), (X, Y, Z, C) in zip(coordinates, points3d):
    print(x,'\t', y,'\t', X,'\t', Y,'\t', Z)
    A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x*X, x*Y, x*Z, x])
    A.append([ 0, 0, 0, 0,-X, -Y, -Z, -1, y*X, y*Y, y*Z, y])

A = np.array(A)
print(A)

print("computing SVD")

u, s, vh = np.linalg.svd(A)

print(u)
print(s)
print(vh)

p = vh[:,-1].reshape((3,4))

points2d = list()
for X in points3d:
    X = np.array(X)
    r = np.matmul(p,X)
    s1 = r[0]/r[2]
    s2 = r[1]/r[2]
    print('#'*50)
    print(s1,s2)
    points2d.append(np.matmul(p,X))



# = np.matmul(p, points3d)
points2d = np.array(points2d)
coordinates = np.array(coordinates[:13])
print('*'*30)
print(points2d)
print(coordinates - points2d)
print('program terminated')
    


plt.plot(points2d[:,0], points2d[:,1],'ro')

plt.plot(coordinates[:,0], coordinates[:,1],'ro')