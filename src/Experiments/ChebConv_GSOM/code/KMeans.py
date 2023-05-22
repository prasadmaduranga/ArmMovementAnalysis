from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feature_encode_file = '../data/encoded_features_fclayer_all.csv'
gsom_coordinates_file = '../results/gsom_out_alldata_train_100_smooth_50_sf_0_83_radius_2.csv'
kmeans_labels = '../data/kmeans_labels_all.csv'

data_filename = "./zoo.txt".replace('\\', '/')



def kmeans_features():
    df = pd.read_csv(feature_encode_file)
    print(df.shape)
    data_training = df.iloc[:, :-1]
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_training)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_training)

    df['label'] = kmeans.labels_
    df.to_csv(kmeans_labels)

def kmeans_gsom_coordinates():
    df = pd.read_csv(gsom_coordinates_file)
    print(df.shape)
    data_training = df.iloc[:, 3:]
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_training)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=8, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_training)

    color_map=['r','g','b','y']

    for i in range(8):
        plt.scatter(data_training.values[kmeans.labels_==i,0],data_training.values[kmeans.labels_==i,1],cmap='coolwarm')
        plt.scatter(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,1],marker='*',s=200,c='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('K-Means on GSOM map coordinates')

    plt.show()



    # df['label'] = kmeans.labels_
    # df.to_csv(kmeans_labels)

if __name__ == '__main__':
    # kmeans_features();
    kmeans_gsom_coordinates();


