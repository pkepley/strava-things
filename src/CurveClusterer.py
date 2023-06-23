import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, pairwise_distances


class CurveClusterer:
    def __init__(self, delta_dist=0.01, min_time=10):
        self.delta_dist = delta_dist
        self.resample_distances = np.arange(0, 1.0, delta_dist)
        self.n_feat = 2 * len(self.resample_distances)
        self.n_half = len(self.resample_distances)
        self.min_time = min_time


    def prep_activity(self, df):
        # only retain whwat we need
        keep_cols = ['distance', 'latitude', 'longitude']
        if df.index.name == 'time':
            df = df.reset_index()
        df_tmp = df[keep_cols]
        df_tmp = df[df['time'] >= self.min_time].reset_index()

        # remove the distance that we truncated, and convert to [0, 1]
        df_tmp['distance'] = df_tmp['distance'] - df_tmp['distance'].min()
        df_tmp['distance'] = df_tmp['distance'] / df_tmp['distance'].max()

        # build interpolators on the [0, 1] interval
        f_lats = interp1d(df_tmp['distance'], df_tmp['latitude'])
        f_lons = interp1d(df_tmp['distance'], df_tmp['longitude'])

        # re-sample the data to the desired sampling
        lons_rsmpl = f_lons(self.resample_distances)
        lats_rsmpl = f_lats(self.resample_distances)

        # pack into required format for feature matrix / sklearn
        X_rsmpl = np.zeros((1, self.n_feat))
        X_rsmpl[0, :self.n_half] = lons_rsmpl
        X_rsmpl[0, self.n_half:] = lats_rsmpl

        return X_rsmpl


    def prep_all_activities(self, dfs):
        n_activities = len(dfs)
        X_feature =  np.zeros((n_activities, self.n_feat))

        for i, df in enumerate(dfs):
           X_feature[i, :]= self.prep_activity(df).ravel()

        return X_feature


    def fit(self, df_ex, prepped=False, min_n_cluster=2, max_n_cluster=None, verbose=False):
        # prep the features
        if not prepped:
            X_feature = self.prep_all_activities(df_ex)
        else:
            X_feature = df_ex

        self.n_train = X_feature.shape[0]

        # get pair-wise "distances" over the lat/lon/[0,1] sampling
        # will use these in an Agglomerative Clusterer
        dd = pairwise_distances(X_feature)
        np.fill_diagonal(dd, 0)

        # if max_n_cluster is not provided, use a (dumb) heuristic
        # to guess at a max feasible value...
        # TODO: this is very very dumb and probably wasteful!
        #       this should be revised
        if max_n_cluster is None:
            max_n_cluster = self.n_train // 2
        if max_n_cluster < min_n_cluster:
            max_n_cluster = min_n_cluster

        # we will use mean silhouette values to determine the best
        # fit
        sv_best = -np.inf
        hc_best = AgglomerativeClustering()

        for k in range(min_n_cluster, max_n_cluster):
            # cluster
            hc = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average"
            )
            hc.fit(dd)
            lbls_curr = hc.labels_

            # get silhouette scores
            sil_vals = silhouette_samples(
                dd, lbls_curr, metric='precomputed'
            )
            sv_curr = sil_vals.mean()

            if verbose:
                print(f"{k} cluster silhouette score: {sv_curr}")

            if sv_curr > sv_best:
                if verbose:
                    print(f"silouhette score improved @ {k}: " +
                          f"{sv_curr:0.02f} > {sv_best:0.02f}")
                k_best  = k
                hc_best = hc
                sv_best = sv_curr
                lbls_best = hc_best.labels_

        # save clusterer
        self.n_cluster = k_best
        self.hc = hc_best

        # get the means of each cluster
        X_feature_means = np.zeros((k_best, self.n_feat))
        for lbl in range(k_best):
            X_feature_means[lbl, :] = (
                X_feature[lbls_best == lbl, :].mean(axis=0).ravel()
            )
        self.cluster_means = X_feature_means

        # compute mean squared distances with clusters
        self._between_cluster_mean_dist = np.zeros(self.n_cluster)
        self._within_cluster_mean_dist = np.zeros(self.n_cluster)

        for lbl in range(k_best):
            # get the mean distance between current cluster's members
            # and all of the cluster mean curves
            cluster_mean_dist = cdist(
                X_feature[lbls_best == lbl, :],
                X_feature_means
            ).mean(
                axis = 0
            )

            # mean distance beween current cluster's members
            # and the current cluster's mean curve
            self._within_cluster_mean_dist[lbl] = (
                cluster_mean_dist[lbl]
            )

            # mean of mean distances beween current cluster's members
            # and every other cluster's mean curve
            jjj = np.arange(k_best) != lbl
            self._between_cluster_mean_dist[lbl] = (
                cluster_mean_dist[jjj].mean()
            )

        # convert the clusterer to a KMeans clusterer
        # (so we don't have to retain the distance matrix!)
        self.kmeans = KMeans(
            n_clusters=self.n_cluster,
            init=self.cluster_means,
            n_init=1,
            max_iter=1
        )
        self.kmeans.fit(X_feature_means)

        return self.kmeans


    def fit_predict(self, df_ex, prepped=False, max_n_cluster=None, verbose=False):
        if prepped:
            X_feature = df_ex
        else:
            X_feature = self.prep_all_activities(df_ex)

        self.fit(X_feature, prepped=True, max_n_cluster=max_n_cluster,
                 verbose=verbose)

        return self.kmeans.predict(X_feature)


    def predict(self, df_ex, prepped=False):
        if prepped:
            X_feature = df_ex
        else:
            X_feature = self.prep_all_activities(df_ex)

        return self.kmeans.predict(X_feature)

    def refit_required(self, df_new, badness_threshold=0.5,
                        prepped=False):
        if prepped:
            X_feature = df_new
        else:
            X_feature = self.prep_all_activities(df_new)

        labels = self.kmeans.predict(X_feature)
        distances_to_centers = cdist(X_feature, self.cluster_means)

        distance_to_assigned = np.array([
            distances_to_centers[i, k] for i, k in enumerate(labels)
        ])

        # check how far each new curve is f
        ii = np.arange(self.n_cluster)
        distance_to_others = np.array([
            distances_to_centers[i, ii != k].mean()
            for i, k in enumerate(labels)
        ])

        # bad assignments may have similar mean distances
        # within and without their assigned clusters
        fraction_assigned_over_other = (
            distance_to_assigned / distance_to_others
        )

        # bad assignments may have distances to their assigned
        # clusters that are on par with distances between clusters
        fraction_of_between_mean = (
            distance_to_assigned / self._between_cluster_mean_dist.mean()
        )

        return (
            np.any(fraction_assigned_over_other > badness_threshold) or
            np.any(fraction_of_between_mean > badness_threshold)
        )
