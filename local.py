#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for multi-shot ridge
regression with decentralized statistic calculation
"""
import json
import os
import sys

import numpy as np

import regression as reg
from lr_parser import gradient, iris_parser, loss


def local_0(args):
    input_list = args["input"]
    lamb = input_list["lambda"]

    X, y = iris_parser(args)
    beta_vec_size = X.shape[1]

    np.save(os.path.join(args['state']['cacheDirectory'], 'X.npy'), X)
    np.save(os.path.join(args['state']['cacheDirectory'], 'y.npy'), y)

    output_dict = {
        "beta_vec_size": beta_vec_size,
        "computation_phase": "local_0"
    }

    cache_dict = {
        "covariates": 'X.npy',
        "dependents": 'y.npy',
        "lambda": lamb,
    }

    computation_output_dict = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_output_dict)


def local_1(args):
    X, y = iris_parser(args)
    w = args["input"]["remote_beta"]
    w = np.squeeze(w)

    grad = gradient(X, y, w)
    obj_val = loss(X, y, w)
    
    output_dict = {
        "local_grad": grad.tolist(),
        "obj_val": obj_val,
        "computation_phase": "local_1"
    }

    cache_dict = {}

    computation_phase = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_phase)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_0' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
