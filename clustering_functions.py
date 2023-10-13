#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import xesmf as xe
import seaborn as sns

import cartopy

from math import ceil

import climpred

from eofs.xarray import Eof

from scipy.stats import boxcox
from sklearn.cluster import KMeans


# pre-processing functions

def calculate_anomalies(x):
    return(x-x.mean(dim='time'))

def filter_morocco(dataset):
    return(dataset.sel(latitude=slice(36,30), longitude=slice(-11,0)))

def filter_mediterranean(dataset):
    return(dataset.sel(latitude=slice(25, 50), longitude=slice(-20,45)))


def preprocess_dataset(filename, variable_name, multiplication_factor, 
                       geographical_filter, months_filter, anomalies, normalization,
                       rolling_window):
    
    dataset = xr.open_dataset(filename)[variable_name]*multiplication_factor
    
    if geographical_filter=='mediterranean':
        dataset = filter_mediterranean(dataset)
    elif geographical_filter=='morocco':
        dataset = filter_morocco(dataset)
    else:
        print('Geographical filter not recognized, no filter applied')
        
    dataset = dataset.sel(time=np.isin(dataset.time.dt.month, months_filter))
    
    if anomalies==True:
        dataset = dataset.groupby('time.dayofyear').map(calculate_anomalies)
    if normalization==True:
        dataset = dataset/dataset.std(dim='time')
    if rolling_window!=0:
        dataset = dataset.rolling(time=5, min_periods=1, center=True).mean()
        
    return(dataset)


def pr_boxcox_transformation(pr_dataset):
    
    transformed_pr = pr_dataset.values
    transformed_pr[transformed_pr < 0] = 0
    transformed_pr = transformed_pr + 0.00000001
    
    if np.ndim(pr_dataset)==1:
        
        transformed_pr = boxcox(transformed_pr)[0]
        
    else:

        for i in range(pr_dataset.shape[1]):
            for j in range(pr_dataset.shape[2]):
        
                transformed_pr[:, i, j] = boxcox(transformed_pr[:, i, j])[0]


    transformed_pr = (transformed_pr - np.mean(transformed_pr, axis=0))/np.std(transformed_pr, axis=0)
    
    return(transformed_pr)
    

# clustering and dimensionality reduction functions

def reshape_data_for_clustering(xarray_data):
    
    # extract numpy array from xarray object
    data = xarray_data.values
    
    # reshape
    nt,ny,nx = data.shape
    data = np.reshape(data, [nt, ny*nx], order='F')
    
    return(data)


def eof_analysis(dataset_xarray, pc_number):
    
    solver = Eof(dataset_xarray, center=True)
    
    eofs = solver.eofs()
    variance_fraction = solver.varianceFraction()
    pcs = solver.pcs(npcs=pc_number)
    
    return(eofs, variance_fraction, pcs)


def get_cluster_fraction(m, label):        
        return (m.labels_==label).sum()/(m.labels_.size*1.0)
    
def get_average_cluster_size(m, cluster_number):
    
    for i in range(m.n_clusters):
        
        (m.labels_==cluster_number).sum()


def calculate_regime_length(labels):

    j=1
    l=labels[0]

    lengths = []

    for i in range(len(labels)-1):

        if(labels[i+1]==labels[i]):
            j=j+1

        else:
            lengths.append(pd.DataFrame(data={"Regime": l,"Length": j}, index=[0]))
            l = labels[i+1]
            j=1

    lengths_df = pd.concat(lengths) 
    return(lengths_df)    



def calculate_threshold_matrix(pr_dataset, quantile_number, threshold_type):
    
    # convert rainfall vector into 1-0 based on exceedance of specified
    if(threshold_type=='higher'):
        threshold_matrix = xr.where(pr_dataset>pr_dataset.quantile(quantile_number, dim='time'), 1, 0)
    elif(threshold_type=='lower'):
        threshold_matrix = xr.where(pr_dataset<pr_dataset.quantile(quantile_number, dim='time'), 1, 0)
    else:
        print('invalid threshold type')
        
    return(threshold_matrix)




##### ----- cluster processing -----------

def reshape_centroids_pca_kmeans(kmeans, eofs, pc_number):
    
    centers = []

    for i in range(kmeans.n_clusters):
        

        onecen = kmeans.cluster_centers_[i][0]*eofs[0, :, :]

        for j in range(pc_number-1):
                onecen = onecen + kmeans.cluster_centers_[i][j+1]*eofs[j+1, :, :]

        centers.append(onecen)

    centers_xr = xr.concat(centers, dim='label')
    return(centers_xr)

def assign_labels(xarray_array, labels):
    
    output = xarray_array.assign_coords(label=("time", labels))
    
    return(output)

def calculate_clusters(xarray_data, cluster_number, calculation_steps = 50):
    
    # reshape data
    data = reshape_data_for_clustering(xarray_data)

    # calculate clusters
    mk = KMeans(n_clusters=cluster_number, n_init=calculation_steps, random_state=0).fit(data)
    
    return(mk)

def reshape_centroids_kmeans(kmeans, dataset_xarray):
    
    k = kmeans.n_clusters
    nt,ny,nx =dataset_xarray.values.shape

    centroids = kmeans.cluster_centers_.reshape(k, ny,nx, order='F')
    
    centroids_xr = xr.DataArray(centroids, coords=dataset_xarray[0:k, :, :].coords, 
                         dims=dataset_xarray[0:k, :, :].dims, attrs=dataset_xarray[0:k, :, :].attrs)
    
    return(centroids_xr)
    




##### ----- visualisation functions -----------

def visualise_contourplot(dataset_xarray, unit, cluster_results, cluster_order,
                          regime_names, vmin, vmax, steps, color_scheme, 
                          col_number=2,borders=True, projection=ccrs.Orthographic(0,45)):
    
    nt,ny,nx = dataset_xarray.values.shape
    x,y = np.meshgrid(dataset_xarray.longitude, dataset_xarray.latitude)

    proj = projection
    fig, axes = plt.subplots(1,col_number, figsize=(14, 7), subplot_kw=dict(projection=proj))

    regimes = regime_names

    for i in range(nt):
        
        cluster=cluster_order[i]
        cs = axes.flat[i].contourf(x, y, dataset_xarray[cluster, :, :],
                                   levels=np.arange(vmin, vmax, steps), 
                                   transform=ccrs.PlateCarree(),
                                   cmap=color_scheme)
        axes.flat[i].coastlines()
        
        if borders==True:
            axes.flat[i].add_feature(cartopy.feature.BORDERS)
            
        title = '{}, {:4.1f}%'.format(regimes[i], get_cluster_fraction(cluster_results, cluster)*100)
        axes.flat[i].set_title(title, fontsize=16)
    plt.tight_layout()
    return(fig)


def visualise_spatial_oddsratio(dataset_xarray, unit, cluster_order, color_scheme, vmin, vmax, steps, 
                                title, regime_names, borders=True, projection=ccrs.PlateCarree(central_longitude=0), col_number=8):
    
    nt,ny,nx = dataset_xarray.values.shape
    
    proj=projection
    
    regimes = regime_names
    
    fig, axes = plt.subplots(1,col_number, figsize=(14, 7), subplot_kw=dict(projection=proj))
    
    for i in range(nt):
        
        cluster=cluster_order[i]
        cs = dataset_xarray[cluster, :, :].plot(ax=axes.flat[i], colors=color_scheme,
                                          transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, levels=steps,
                                         add_colorbar=False)
        axes.flat[i].coastlines()
        
        if borders==True:
            axes.flat[i].add_feature(cartopy.feature.BORDERS)
            
        sub_title = '{}'.format(regimes[i])
        axes.flat[i].set_title(sub_title, fontsize=16)

    plt.tight_layout()
    return(fig)





##### ----- evaluation functions -----------

def calculate_acc(cluster_centers1, cluster_centers2, k):
    
    columns = []
    
    for i in range(0,k):
    
        rows = []
        
        for j in range(0,k):
        
            metrics_test = climpred.metrics._pearson_r(forecast = cluster_centers1[i, :,:], 
                                                   verif = cluster_centers2[j, :,:])
        
            rows.append(metrics_test.values)
        
        columns.append(max(np.absolute(rows)))

    acc = min(columns)
    return(acc)


def calculate_conditional_probability_change(threshold_matrix, kmeans, comparison, shift_value=0):
    
    # add cluster assignment to threshold vector
    threshold_matrix_label = threshold_matrix.assign_coords(label=("time", np.roll(kmeans.labels_, shift_value)))

    # probability conditional on weather type
    n_wr = threshold_matrix_label.groupby('label').mean()

    # overall probability
    n_total = threshold_matrix_label.mean(dim='time')
    
    if comparison=='difference':
        ds = n_wr - n_total
    elif comparison=='ratio':
        ds = n_wr/n_total
    elif comparison=='none':
        ds = n_wr
    else:
        print('invalid entry for diff_or_quot')
    
    return(ds)
