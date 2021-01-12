import sys
from skimage.io import imread,imsave,imshow
from sklearn.cluster import KMeans

file=sys.argv[1]
k=int(sys.argv[2])

prev=imread(file)
arr = prev.reshape((-1, 3))
kmeans = KMeans(n_clusters=k, random_state=0).fit(arr)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
x = centers[labels]
quant= x.reshape(prev.shape)
imsave(f'{file[:-4]}_{k}.png',quant)
