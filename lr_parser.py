#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:28:50 2018

@author: Harshvardhan
"""

import os

import numpy as np
np.seterr(all='ignore')


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def iris_parser(args):
    data = np.genfromtxt(os.path.join(args['state']['baseDirectory'],
                                      'data.csv'),
                         delimiter=',')
    X = data[:, :2]
    X = add_intercept(X)
    y = data[:, -1]
    y = (y != 0) * 1

    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(X, y, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return np.dot(X.T, (h - y)) / y.size


def loss(X, y, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
