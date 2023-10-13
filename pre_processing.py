#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:20:12 2023

@author: fionaspuler
"""

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns

from global_land_mask import globe

from scipy.stats import boxcox
import random

import clustering_functions as cf

from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression

g0 = 9.80665
extended_winter_months = [11, 12, 1, 2, 3]
years = list(range(1940, 2023))

precip_colors = ['#F5F5F5', '#C7EAE5', '#80CDC1', '#35978F', '#01665E', '#003C30']


# import and pre-process z500 data

z500 = cf.preprocess_dataset(filename = 'data.nosync/era5_z500_daily_250_atlantic_1940_2022.nc',
                              variable_name = 'z', 
                              multiplication_factor = 1/g0, 
                              geographical_filter = 'mediterranean', 
                              months_filter = extended_winter_months, 
                              anomalies = True, 
                              normalization = True,
                              rolling_window = 5)

# import and pre-process precipitation

pr = cf.preprocess_dataset(filename = 'data.nosync/era5_pr_daily_mr_1940_2022.nc',
                              variable_name = 'tp', 
                              multiplication_factor = 1000, 
                              geographical_filter = 'morocco', 
                              months_filter = extended_winter_months, 
                              anomalies = False, 
                              normalization = False,
                              rolling_window = 3)

# aggregate precipitation spatially

pr_spatial = pr.mean(dim=['latitude', 'longitude'])

# transform precipitation to normal distribution

pr_boxcox = cf.pr_boxcox_transformation(pr)
pr_boxcox_spatial = cf.pr_boxcox_transformation(pr_spatial)

pr_boxcox_spatial_xr = xr.DataArray(pr_boxcox_spatial, coords=pr[:, 0, 0].coords, 
                         dims=pr[:, 0, 0].dims, attrs=pr[:, 0, 0].attrs)
pr_boxcox_xr = xr.DataArray(pr_boxcox, coords=pr.coords, 
                         dims=pr.dims, attrs=pr.attrs)

# calculate 95th percentile threshold matrix

threshold_qn95 = cf.calculate_threshold_matrix(pr_dataset = pr, 
                                            quantile_number = 0.95, 
                                            threshold_type = 'higher')

# calculate precipitation occurrence matrix

occurrence_matrix = xr.where(pr>1, 1, 0)

# calculate precipitation clusters

pr_clusters = cf.calculate_clusters(xarray_data = pr, cluster_number = 6, calculation_steps=100)
pr_centroids = cf.reshape_centroids_kmeans(pr_clusters, pr)

# calculate mean precipitation per cluster

pr_spatial_labeled = cf.assign_labels(pr_spatial, pr_clusters.labels_)
mean_pr_per_wr = pr_spatial_labeled.groupby('label').mean().to_dataframe()
mean_pr_per_wr

z500_clusters_kmeans = cf.calculate_clusters(xarray_data = z500, cluster_number = 5)
z500_centroids_kmeans = cf.reshape_centroids_kmeans(z500_clusters_kmeans, z500)

z500_clusters_kmeans8 = cf.calculate_clusters(xarray_data = z500, cluster_number = 8)
z500_centroids_kmeans8 = cf.reshape_centroids_kmeans(z500_clusters_kmeans8, z500)

nt,ny,nx = z500.values.shape
z500_reshaped = np.reshape(z500.values, [nt, ny*nx], order='F')






