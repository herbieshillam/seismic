from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# Importing dataset - you will have to change this to match the path on your computer

italydata = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Spyder/Main Italy Data.xlsx')
data = italydata.parse('Sheet1')
data.head()

# Importing the outline:

outline = pd.ExcelFile('/Users/Herbie/Documents/Uni/Masters/Mathematical and Data Modelling/Book6.xlsx')

italyoutline = outline.parse('Sheet1')
italyoutline.head()
italyoutline.columns = ['A', 'B','C','D']

# Filling NaN values with mean:

data = data.fillna(data.mean())

# Normalising Depth:

x = data[['Depth']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
depth_scaled = min_max_scaler.fit_transform(x)
depth_normalised = pd.DataFrame(depth_scaled)
data['Depth Norm']= depth_normalised

# Normalising Magnitude:

x = data[['Mom-Mag Mw']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
mag_scaled = min_max_scaler.fit_transform(x)
mag_normalised = pd.DataFrame(mag_scaled)
data['Mag Norm']= mag_normalised

# Normalising Days Passed:

x = data[['Days Passed + max negative value']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
days_passed_scaled = min_max_scaler.fit_transform(x)
days_passed_normalised = pd.DataFrame(days_passed_scaled)
data['Days Passed Since 01/01/1900 Norm']= days_passed_normalised

# Clustering:

k = 4 # Can change this
clusteringdata =  data[['Depth Norm', 'Mag Norm', 'Days Passed Since 01/01/1900 Norm']]
kmeans = KMeans(n_clusters=k)
kmeans.fit(clusteringdata)

print(kmeans.cluster_centers_)
print(kmeans.labels_)

#plt.figure()     # No point in this since now clustering based on 3 variables
#plt.scatter(data['Depth Norm'],data['Mag Norm'], c=kmeans.labels_, cmap='rainbow',s=6)
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')

plt.figure()
for i in range(0, k ):
    plt.scatter(data.loc[kmeans.labels_ == i]['Longitude'], data.loc[kmeans.labels_ == i]['Latitude'],s=4)
    plt.legend(loc="upper right")
    plt.legend(['Cluster 1', 'Cluster 2','Cluster 3', 'Cluster 4'])
    
# Plotting the outline:

plt.plot(italyoutline['B'],italyoutline['A'],'k')
plt.plot(italyoutline['D'],italyoutline['C'],'k')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters plotted on map of Italy')

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

data['Cluster number'] = kmeans.labels_