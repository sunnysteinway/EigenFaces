import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

def open_npy_file(strPath):
    """open .npy file with given filepath"""
    filepath = Path(strPath)    # make the string path as a filepath
    return np.load(filepath)

# load files
strPath = r'C:\Users\neilp\Documents\VSCode\SVD\neutralFaces.npy'  # create a string path for the filename
neutralFaces = open_npy_file(strPath)
strPath = r'C:\Users\neilp\Documents\VSCode\SVD\happyFaces.npy'  # create a string path for the filename
happyFaces = open_npy_file(strPath)

# combine neutral faces & happy faces together
allFaces = np.concatenate((neutralFaces, happyFaces), axis=1)

# create Xs and Ys
x = allFaces.T  # n * points
y = np.concatenate((np.zeros([1, neutralFaces.shape[1]], dtype=np.byte), 
    np.ones([1, happyFaces.shape[1]], dtype=np.byte)), axis=1)
y = y.T # n * 1

# split the data into training set and testing set
# x: n * points
# y: n * 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=142)

# calculate the average face
avgFace = np.mean(x_train.T, axis=1, dtype=np.float64)

# calculate the SVD
n = x_train.shape[0]
m = x_train.shape[1]
phi = np.empty([m, n])

for i in range(0, n):
    phi[:, i] = x_train[i, :] - avgFace

U, S, VH = np.linalg.svd(phi, full_matrices=False)

# set number of testing data
no = 100

# set number of significant eigen vectors
pc = 3

# project onto face space
alphaNeutral = np.matmul(U[:, 0:pc].T, neutralFaces[:, 0:no])  # pc * no
alphaHappy   = np.matmul(U[:, 0:pc].T, happyFaces  [:, 0:no])  # pc * no

x0 = np.vsplit(alphaNeutral, 3)[0]
y0 = np.vsplit(alphaNeutral, 3)[1]
z0 = np.vsplit(alphaNeutral, 3)[2]
x1 = np.vsplit(alphaHappy  , 3)[0]
y1 = np.vsplit(alphaHappy  , 3)[1]
z1 = np.vsplit(alphaHappy  , 3)[2]

fig = plt.figure()
plt.suptitle("Scatter Plot between Neutral and Happy")
ax = fig.add_subplot(projection='3d')
ax.scatter(x0, y0, z0, marker='*', c='r')
ax.scatter(x1, y1, z1, marker='<', c='m')
ax.set_xlabel("Eigen Vector #1")
ax.set_ylabel("Eigen Vector #2")
ax.set_zlabel("Eigen Vector #3")
plt.show()

# classification

tst = 100   # number of data shall be drawn

rslt = [-1] * tst
ans = y_test[0:tst, 0].T.tolist()
for t in range(0, tst):

    # set errors to zero
    e0 = 0
    e1 = 0

    # project testing data onto two face spaces
    alpha0 = np.matmul(U[:, 0:pc].T, x_test[t, :].T)
    alpha1 = np.matmul(U[:, 0:pc].T, x_test[t, :].T)
    
    # calculate MSEs
    for i in range(0, no):
        e0 += np.linalg.norm(alpha0 - alphaNeutral[:, i])
        e1 += np.linalg.norm(alpha1 - alphaHappy  [:, i])
    if e0 < e1:
        rslt[t] = 0
    else:
         rslt[t] = 1

cntNeutral = 0
cntHappy   = 0

errorNeutral = 0
errorHappy   = 0

# identify how many neutrals & happies
for i in range(0, tst):
    if ans[i] == 0:
        cntNeutral += 1
    elif ans[i] == 1:
        cntHappy += 1

# calculate errors
for i in range(0, tst):
    
    if ans[i] != rslt[i]:
        if ans[i] == 0:
            errorNeutral += 1
        elif ans[i] == 1:
            errorHappy += 1

totalError = errorNeutral + errorHappy

print("No. of Neutrals:{}".format(cntNeutral))
print("No. of Happies: {}\n".format(cntHappy))

print("Error (total):   {}; Percentage: {}".format(totalError, (totalError / tst)))
print("Error (Neutral): {}".format(errorNeutral))
print("Error (Happy):   {}".format(errorHappy))
