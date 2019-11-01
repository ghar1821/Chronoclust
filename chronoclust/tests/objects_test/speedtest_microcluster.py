#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:04:19 2019

@author: givanna
"""


import timeit
import numpy as np

from objects.microcluster import Microcluster as NewMC
from objects.old_microcluster import Microcluster as OldMC



def test_update_preferred_dimensions_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)
    for p in points:
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.25, 15)


def test_update_preferred_dimensions_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.25, 15)

def test_add_new_points_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)

def test_add_new_points_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)

def test_set_centroid_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)
        mc.set_centroid()

def test_set_centroid_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0, update_centroid=False)
        mc.set_centroid()

def test_get_projected_distance_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add dummy point.
    # the code will break if we try to calculate projected distance to new cluster with no point, which is expected.
    # This is because there should not be any MC with no points in it.
    new_pt = np.random.rand(1, num_dim)[0]
    mc.add_new_point(new_pt, 0)
    mc.update_preferred_dimensions(0.25, 15)

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.get_projected_dist_to_point(p)

def test_get_projected_distance_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add dummy point
    # the code will break if we try to calculate projected distance to new cluster with no point, which is expected.
    # This is because there should not be any MC with no points in it.
    new_pt = np.random.rand(1, num_dim)[0]
    mc.add_new_point(new_pt, 0)
    mc.update_preferred_dimensions(0.25, 15)

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.get_projected_dist_to_point(p)

def test_calculated_radius_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)
    mc.add_new_point(points[0], 0)
    mc.update_preferred_dimensions(0.25, 15)

    for p in points[1:]:
        mc.calculate_projected_radius_squared()
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.25, 15)



def test_calculated_radius_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)
    mc.add_new_point(points[0], 0)
    mc.update_preferred_dimensions(0.25, 15)

    for p in points[1:]:
        mc.calculate_projected_radius_squared()
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.25, 15)

def test_clone_mc_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.get_copy()

def test_clone_mc_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.get_copy()

def test_is_core_old():
    num_dim = 20
    mc = OldMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.05,15)
        mc.is_core(0.1,2,2)

def test_is_core_new():
    num_dim = 20
    mc = NewMC(cf1=np.zeros(num_dim), cf2=np.zeros(num_dim))

    # add 1000 points
    points = np.random.rand(1000, num_dim)

    for p in points:
        mc.add_new_point(p, 0)
        mc.update_preferred_dimensions(0.05,15)
        mc.is_core(0.1,2,2)

if __name__ == '__main__':
    print(timeit.timeit(test_update_preferred_dimensions_old, number=1000))
    print(timeit.timeit(test_update_preferred_dimensions_new, number=1000))

    print(timeit.timeit(test_add_new_points_old, number=1000))
    print(timeit.timeit(test_add_new_points_new, number=1000))

    print(timeit.timeit(test_set_centroid_old, number=1000))
    print(timeit.timeit(test_set_centroid_new, number=1000))

    print(timeit.timeit(test_get_projected_distance_old, number=1000))
    print(timeit.timeit(test_get_projected_distance_new, number=1000))

    print(timeit.timeit(test_calculated_radius_old, number=1000))
    print(timeit.timeit(test_calculated_radius_new, number=1000))

    print(timeit.timeit(test_clone_mc_old, number=1000))
    print(timeit.timeit(test_clone_mc_new, number=1000))

    print(timeit.timeit(test_is_core_old, number=1000))
    print(timeit.timeit(test_is_core_new, number=1000))
