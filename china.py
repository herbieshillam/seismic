# China

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# Importing dataset - you will have to change this to match the path on your computer

china_data = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Spyder/China/China excel.xlsx')
china_data = china_data.parse('Sheet2')
china_data.head()

# Filling NaN values with mean:

china_data = china_data.fillna(china_data.mean())

# Normalising Depth:

x = china_data[['depth']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
depth_scaled = min_max_scaler.fit_transform(x)
depth_normalised = pd.DataFrame(depth_scaled)
china_data['Depth Norm']= depth_normalised

# Normalising Magnitude:

x = china_data[['mag']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
mag_scaled = min_max_scaler.fit_transform(x)
mag_normalised = pd.DataFrame(mag_scaled)
china_data['Mag Norm']= mag_normalised

# Clustering:

k = 4 # Can change this
clusteringdata = china_data[['Depth Norm', 'Mag Norm']]
kmeans = KMeans(n_clusters=k)
kmeans.fit(clusteringdata)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

# Cities:

china_all_data = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Spyder/china/China excel.xlsx')
china_all_data = china_all_data.parse('Sheet3')
china_all_data.head()

fig, ax = plt.subplots()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Clusters plotted on China Map')

ax.scatter(china_all_data['long'], china_all_data['lat'],s=2, label='Cities & Coastline')

###

for i in range(0, k ):
    ax.scatter(china_data.loc[kmeans.labels_ == i]['longitude'], \
               china_data.loc[kmeans.labels_ == i]['latitude'], \
               s=6, label = 'Cluster ' + str(i + 1))      
 
ax.legend(loc = 'lower left', prop={'size': 9})
#ax.set_ylim(10,60)
#ax.set_xlim(60, 140)
#['Cities and Coastline','Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4'])
#ax.legend(loc="lower left")