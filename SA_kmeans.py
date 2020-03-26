from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# Importing dataset - you will have to change this to match the path on your computer

SA_data = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Spyder/south america/SA_data.xlsx')
SA_data = SA_data.parse('Sheet1')
SA_data.head()

# Filling NaN values with mean:

SA_data = SA_data.fillna(SA_data.mean())

# Normalising Depth:

x = SA_data[['depth']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
depth_scaled = min_max_scaler.fit_transform(x)
depth_normalised = pd.DataFrame(depth_scaled)
SA_data['Depth Norm']= depth_normalised

# Normalising Magnitude:

x = SA_data[['mag']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
mag_scaled = min_max_scaler.fit_transform(x)
mag_normalised = pd.DataFrame(mag_scaled)
SA_data['Mag Norm']= mag_normalised

# Clustering:

k = 4 # Can change this
clusteringdata = SA_data[['Depth Norm', 'Mag Norm']]
kmeans = KMeans(n_clusters=k)
kmeans.fit(clusteringdata)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

SA_all_data = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Spyder/south america/worldcities.xlsx')
SA_all_data = SA_all_data.parse('Sheet1')
SA_all_data.head()

plt.figure()
plt.scatter(SA_all_data['latitude'], SA_all_data['longitude'],s=2)
for i in range(0, k ):
    plt.scatter(SA_data.loc[kmeans.labels_ == i]['longitude'], SA_data.loc[kmeans.labels_ == i]['latitude'],s=6,label="aa")
    #plt.legend(loc="upper right")
    #plt.legend(['Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4', 'Cities + Coastline'])

#plt.scatter(SA_all_data['latitude'], SA_all_data['longitude'],s=2) #labelled wrong on excel
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters plotted on South America Map')
plt.legend(loc="lower right")
plt.legend([ 'Cities + Coastline','Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4'])


# Elbow plot:
    
distortions = []
K = range(1,10)
for k in K:
    
    kmeanModel = KMeans(n_clusters=k).fit(clusteringdata)
    kmeanModel.fit(clusteringdata)
    distortions.append(sum(np.min(cdist(clusteringdata,kmeanModel.cluster_centers_,'euclidean')**2, axis=1)) / clusteringdata.shape[0])
    
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')

# Adding column to data identifying which cluster each is in

SA_data['Cluster number'] = kmeans.labels_