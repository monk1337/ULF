from sklearn.cluster import KMeans

class KmeansClustering(object):

    def __init__(
                self,
                dataframe, 
                n_clusters,
                random_state=0,
                max_iter = 300
                ):

        self.dataframe = dataframe
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter


    def kmeans_model(self):

        kmeans_ = KMeans(n_clusters = self.n_clusters, 
                         algorithm='auto', 
                         init='k-means++', 
                         max_iter = self.max_iter,
                         random_state = self.random_state).fit(self.dataframe)
        clusters = kmeans_.predict(self.dataframe)

        return clusters