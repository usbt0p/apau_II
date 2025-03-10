'''Implementation of the K-Means algorithm in Python. As a demonstration and test, 
we will use the same synthetic dataset used in the Jupyter notebook seen in class.

### Features:

    - For centroid initialization, it is possible to choose between:
        Random initialization of points not necessarily existing, within the bounds of the data
        Random initialization of centroids from existing dataset points
    - Distance measure: Euclidean
    - If a point is equidistant from two centroids, size balance is applied 
        (assignment to the centroid with fewer points)
    - Convergence criterion: should stop after max_iter iterations (to be specified as a parameter)
'''

import matplotlib.pyplot as plt
import random as rand
from sklearn.datasets import make_blobs
import numpy as np


class KMeansFromScratch():
    '''K-means clustering algorithm from scratch.

    Supports user-defined centroid initialization and number of iterations.

    Tie breaking in case of several centroids being at the same distance
    is done by assigning the point to the less populated cluster.

    Written without AI assisted generation.
    '''

    def __init__(self, data: np.ndarray, k: int, n_iter: int, points_as_centroids: bool = True):
        '''Create a KMeansFromScratch object.

        Parameters
        ----------
        data : np.ndarray
            The data to cluster.
        k : int
            The number of clusters.
        n_iter : int
            The number of iterations to run the algorithm.
        points_as_centroids : bool
            Change behavior of centroid initialization.
            Use random points from the data as centroids if True or 
            use random points within the bounds of the data if False.
        '''
        self.data = data  # number of points in
        self.k = k  # number of clusters and therefore number of centroids
        self.n_iter = n_iter  # num of iterations at which to stop
        self.points_as_centroids = points_as_centroids 

        # data points, dimension of the points
        self.n_points, self.n_features = data.shape
        self.centroids = self.select_initial_centroids(
            points_as_centroids)  # initial centroids
        self.cluster_idxs = None  # cluster index of each point

    def select_initial_centroids(self, points_as_centroids: bool, override=True) -> np.ndarray:
        '''Random initialization of centroids from dataset points.
        When executed, it overrides the default initialization done in __init__.

        Parameters
        ----------
        points_as_centroids : bool
            If True, select k random points from the data as centroids.
            If False, select random points within the bounds of the data.
        override : bool
            Defaults to True. Override the previous initialization of centroids upon
            calling this method, or just return the new centroids.

        Returns
        -------
        np.ndarray
            The initial centroids.
        '''
        if points_as_centroids: # select k random points from the data as centroids
            c_idx = rand.sample(range(self.n_points), self.k)
            centroids = self.data[c_idx] # i fucking love numpy indexing
        else:
            # random initialization of centroids within the bounds of the data
            # find bounds of every dimension
            bounds = [(float(X[:, dim].min()), float(X[:, dim].max()))
                      for dim in range(self.n_features)]

            centroids = []  # with te bounds, generate the centroids
            for cntr in range(self.k):
                cntr = [rand.uniform(bound[0], bound[1]) for bound in bounds]
                centroids.append(cntr)
            centroids = np.asarray(centroids)

        if override: self.centroids = centroids   # override previous initiailization
        return centroids

    def get_new_clusters(self, points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        '''Assign each point to the cluster of the closest centroid.
        In case of several centroids being at the same distance, break ties by
        assigning the point to the less populated cluster.

        Parameters
        ----------
        points : np.ndarray
            The data points to cluster.
        centroids : np.ndarray
            The current centroids.

        Returns
        -------
        np.ndarray
            The cluster index of each point
        '''

        new_clusters_idx = []
        # points per cluster, used to reduce computation
        ppc = {k: 0 for k in range(len(centroids))}

        for point in points:

            # find dist from point to each centroid
            # point(n_features, ) - centroids(k, n_features)
            dist: np.ndarray = np.linalg.norm(point - centroids, axis=1)

            # if min dist is equal to more than 1 centroid, assign the one with less points
            min_dist = min(dist)
            # indexes of duped distances
            dupe_dists = [i for i, d in enumerate(dist) if d == min_dist]

            if len(dupe_dists) > 1:  # if duplicate min dists have been found
                closest = min(dupe_dists, key=lambda i: ppc[i])

            else:  # no duped dists, closest idx is correct
                closest = dupe_dists[0]

            new_clusters_idx.append(closest)
            ppc[closest] += 1

        self.cluster_idxs = np.array(new_clusters_idx)
        return np.array(new_clusters_idx)

    def update_centroids(self, cluster_idxs: np.ndarray):
        '''Set new centroid locations to the mean of their clusters.

        Parameters
        ----------
        cluster_idxs : np.ndarray
            The cluster index of each point.

        Returns
        -------
        np.ndarray
            The new centroids.
        '''

        new_centroids = []
        for centroid_idx in range(self.k):
            # cool numpy sheananigans
            points = self.data[cluster_idxs == centroid_idx]
            # axis 0 means columns for some reason
            new_centroids.append(points.mean(axis=0))

        return np.asarray(new_centroids)

    def run(self, debug=False, save_plt=''):
        '''Run the KMeans algorithm. 

        Parameters
        ----------
        debug : bool
            If True, show the clustering process at each iteration.
            Only works for 2D data.
        save_plt : str
            If not empty, save the plots with the given name and format `save_plt + '_{iter}.png'`
            Use the name to specify the path and format of the file.

        ### K-MEANS PSEUDOCODE:
        .. code-block:: python

            init centroids
            while iter != n_iter:
                assign each point to the cluster of the closest centroid
                if all centroids have points:
                    update centroids to be the mean of their elements
                    n_iter += 1
                else: 
                    randomly reinitialize empty centroids

        '''
        # initailization of centroids is done upon instantiation, no need to do it again
        SIZE = 60
        empty_found = False # flag to check if empty centroids were found
        iter = 0

        if debug:
            # show initial centroid location
            plt.scatter(
                k_means.data[:, 0], k_means.data[:, 1], s=SIZE, c=k_means.cluster_idxs)
            plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1],
                        s=SIZE, marker='x', c='r')
            plt.grid(visible=True)
            plt.title('Initial centroids')
            plt.legend(['Data points', 'Centroids'])
            if save_plt: plt.savefig(save_plt + f'_{iter}.png')
            plt.show()

        while iter != self.n_iter:
            new_clusters = self.get_new_clusters(self.data, self.centroids)
            
            # FIXME id ppc is was returned in get_new_clusters, this could be done more efficiently
            if all([len(self.data[new_clusters == i]) > 0 for i in range(self.k)]):  
                new_centroids = self.update_centroids(new_clusters)
                self.centroids = new_centroids  # IMPORTANT: update centroids
                iter += 1

            else:
                # find the empty centroids
                empty_found = True
                empty_centroids = [i for i in range(self.k) 
                                if len(self.data[new_clusters == i]) == 0]
                print(f'{len(empty_centroids)} empty centroids found, reinitializing...')
                
                # generate new centroids for the empty ones, without overriding the rest
                self.centroids[empty_centroids] = self.select_initial_centroids(
                    self.points_as_centroids, override=False)[empty_centroids]

                # now that the empty centroids have been filled, continue as normal

            if debug:
                # show new cluster assignation
                
                plt.scatter(
                    k_means.data[:, 0], k_means.data[:, 1], s=SIZE, c=k_means.cluster_idxs)
                plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1],
                            s=SIZE, c=range(k_means.k), edgecolors='r', linewidths=2)
                plt.grid(visible=True)
                title = f'Results for iteration {iter}' if not \
                    empty_found else 'Empty centroids found, reinitializing...'
                plt.title(title)
                plt.legend(['Data points', 'Centroids'])

                if save_plt: plt.savefig(save_plt + f'_{iter}.png')
                plt.show()
            empty_found = False # reset flag

    def __str__(self):
        d = self.__dict__.copy()
        d['data'] = f"{type(d['data'])} of size {d['data'].size}"
        d['centroids'] = f"{type(d['centroids'])} of size {d['centroids'].size}"
        # remove cluster_idxs from string representation
        del d['cluster_idxs']
        return str(d)


if __name__ == "__main__":
    points = 300
    SIZE = 60
    features = 2
    n_centers = 4
    n_iter = 4

    X, y = make_blobs(  # X are coords of each point, y is the cluster they where generated in
        n_samples=points, n_features=features, centers=n_centers, cluster_std=0.60, random_state=0)

    # DUDA los resultados son MUY diferentes si se inicializan los centroides con puntos del dataset
    k_means = KMeansFromScratch(
        X, n_centers, n_iter, points_as_centroids=False)
    k_means.cluster_idxs = y  # pass centroids for initial debug visualization
    print(k_means)

    # Uncomment and set to False to only show the final clustering
    k_means.run(debug=True, save_plt='') # change save_plt to save the plots
    '''# show new cluster assignation
    plt.scatter(k_means.data[:, 0], k_means.data[:, 1], s=SIZE, c=k_means.cluster_idxs)
    plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1], s=SIZE, c='r', marker='x')
    plt.grid(visible=True)
    plt.show()'''

    ##############################
    #########  TESTING  ##########
    ##############################

    def test_new_cluster_assignations(k_means: KMeansFromScratch):
        '''Test kmeans cluster assignations with 2D graphical representation'''

        inits = k_means.select_initial_centroids(points_as_centroids=True)
        print(type(inits))
        print(inits)

        # show before new cluster assignation
        plt.scatter(X[:, 0], X[:, 1], s=SIZE, c=y)
        plt.scatter(inits[:, 0], inits[:, 1], s=SIZE, c='r', marker='x')
        plt.grid(visible=True)
        plt.show()

        # after closest cluster assignation
        new_clusters = k_means.get_new_clusters(
            k_means.data, k_means.centroids)
        plt.scatter(X[:, 0], X[:, 1], s=SIZE, c=new_clusters)
        plt.scatter(inits[:, 0], inits[:, 1], s=SIZE*2,
                    c=range(k_means.k), edgecolors='r', linewidths=3)
        plt.grid(visible=True)
        plt.show()

    def test_init_centroids(k_means: KMeansFromScratch, select_points: bool):
        '''Test kmeans centroid initialization with 2D graphical representation'''
        assert isinstance(select_points, bool)

        inits = k_means.select_initial_centroids(
            points_as_centroids=select_points)
        print(type(inits))
        print(inits)

        plt.scatter(X[:, 0], X[:, 1], s=10, c=y)
        plt.scatter(inits[:, 0], inits[:, 1], s=50, c='r', marker='x')
        plt.grid(visible=True)
        plt.show()

    def test_assign_points_to_clusters(k_means: KMeansFromScratch):
        '''Test kmeans point with 2D graphical representation'''

        # useful to test tie breaking in behavior
        points = np.asarray([[0, 0], [0, 1], [3, 4], [-1, 3], [0, 0]])
        centroids = np.asarray([[1, 0], [0, 5], [1, 0], [0, 5]])

        res = k_means.get_new_clusters(points, centroids)
        print(res)

    def test_update_centroids(k_means: KMeansFromScratch):
        '''Test centroid updating based on cluster mean.'''

        inits = k_means.select_initial_centroids(points_as_centroids=True)
        print(type(inits))
        print(inits)

        # show before new cluster assignation
        plt.scatter(X[:, 0], X[:, 1], s=SIZE, c=y)
        plt.scatter(inits[:, 0], inits[:, 1], s=SIZE, c='r', marker='x')
        plt.grid(visible=True)
        plt.show()

        # after closest cluster assignation
        new_clusters = k_means.get_new_clusters(
            k_means.data, k_means.centroids)
        plt.scatter(X[:, 0], X[:, 1], s=SIZE, c=new_clusters)
        plt.scatter(inits[:, 0], inits[:, 1], s=SIZE*2,
                    c=range(k_means.k), edgecolors='r', linewidths=3)
        plt.grid(visible=True)
        plt.show()

        # now update centroid locations
        new_centroids = k_means.update_centroids(new_clusters)
        plt.scatter(X[:, 0], X[:, 1], s=SIZE, c=new_clusters)
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1],
                    s=SIZE*2, c=range(k_means.k), edgecolors='r', linewidths=3)
        plt.grid(visible=True)
        plt.show()

    # test_update_centroids(k_means)
    # test_new_cluster_assignations(k_means)
    # test_distance(k_means)
