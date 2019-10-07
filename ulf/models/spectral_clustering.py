from sklearn.cluster import SpectralClustering

class SpectralClustering_(object):

    def __init__(
                self,
                dataframe, 
                n_clusters, 
                random_state= 0,
                assign_labels= 'kmeans'
                ):

        self.dataframe = dataframe
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.assign_labels = assign_labels

    def spectral_model(self):
        
        clustering = SpectralClustering(n_clusters    = self.n_clusters, 
                                        assign_labels = self.assign_labels, 
                                        random_state=self.random_state).fit(self.dataframe)
        return clustering.labels_
        

