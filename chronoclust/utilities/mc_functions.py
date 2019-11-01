#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains all functions required to calculate features of MC.
All of these are implemented using numba for speed.
Created on Fri Aug 16 16:59:24 2019

@author: givanna
"""

from numba import jit
import numpy as np

@jit(nopython=True)
def calculate_squared_variance(cf1, cf2, cum_weight):

    cf2_over_weight = np.divide(cf2, cum_weight)
    cf1_over_weight = np.square(np.divide(cf1, cum_weight))

    squared_variance = np.subtract(cf2_over_weight, cf1_over_weight)

    return squared_variance

@jit(nopython=True)
def update_cf(cf1, cf2, new_point):
    new_cf1 = np.add(cf1, new_point)
    new_cf2 = np.add(cf2, np.square(new_point))

    return new_cf1, new_cf2

@jit(nopython=True)
def calculate_centroid(cf1, cum_weight):
    return np.divide(cf1, cum_weight)

@jit(nopython=True)
def calculate_projected_distance(mc_centroid, mc_pref_dim, new_point):
    point_minus_centroid = np.subtract(new_point, mc_centroid)
    point_minus_centroid_sq = np.square(point_minus_centroid)
    division_result = np.divide(point_minus_centroid_sq, mc_pref_dim)

    distance = np.sum(division_result)

    return distance

@jit(nopython=True)
def calculate_projected_radius_squared(cf1, cf2, pref_dim, cum_weight):
    c2_over_weight = np.divide(cf2, cum_weight)
    c1_over_weight = np.divide(cf1, cum_weight)
    c1_over_weight_sq = np.square(c1_over_weight)

    c2_minus_c1 = np.subtract(c2_over_weight, c1_over_weight_sq)
    c2_minus_c1_over_pdim = np.divide(c2_minus_c1, pref_dim)

    radius_squared = np.sum(c2_minus_c1_over_pdim)

    return radius_squared

@jit(nopython=True)
def clone_cf(cf1, cf2):
    new_cf1 = np.copy(cf1)
    new_cf2 = np.copy(cf2)
    return new_cf1, new_cf2

@jit(nopython=True)
def is_core(cf1, cf2, pref_dim, cum_weight,
            radius_threshold_squared, density_threshold,
            max_subspace_dimensionality):
    radius_squared = calculate_projected_radius_squared(cf1, cf2, pref_dim, cum_weight)

    # the output is a tuple. We only want the count. Hence the weird format.
    # see: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.where.html
    cnt_pdim = len(np.where(pref_dim > 1)[0])

    core_status = (radius_squared <= radius_threshold_squared
            and cum_weight >= density_threshold
            and cnt_pdim <= max_subspace_dimensionality)
    return core_status
