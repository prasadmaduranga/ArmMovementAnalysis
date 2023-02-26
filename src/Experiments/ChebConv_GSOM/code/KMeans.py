from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



feature_encode_file = '../data/encoded_features_fclayer_all.csv'
kmeans_labels = '../data/kmeans_labels_all.csv'
data_filename = "./zoo.txt".replace('\\', '/')


if __name__ == '__main__':

    df = pd.read_csv(feature_encode_file)
    print(df.shape)
    data_training = df.iloc[:, :-1]
    wcss = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
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





    # map_points.to_csv(gsom_output_file, index=False)

    print("complete")