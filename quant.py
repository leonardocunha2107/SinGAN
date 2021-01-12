import 
from skimage.io import imread,imsave,imshow
from sklearn.cluster import KMeans

file=sys.argsv[1]
k=sys.argsv[2]

prev=imread(file)
arr = prev.reshape((-1, 3))
kmeans = KMeans(n_clusters=k, random_state=0).fit(arr)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
x = centers[labels]
quant= x.reshape(prev.shape)
imsave(f'{file[:-3]}.png',quant)
