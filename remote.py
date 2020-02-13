#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys

import numpy as np

import regression as reg

np.seterr(all='ignore')

def remote_0(args):
    """Need this function for performing multi-shot regression"""
    input_list = args["input"]
    first_user_id = list(input_list.keys())[0]
    beta_vec_size = input_list[first_user_id]["beta_vec_size"]

    # Initial setup
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    tol = 2e-8  # 0.01
    eta = 0.01  # 0.05
    count = 0
    loss = 0
    iterations = 10000

    mt, vt = [np.zeros((1, beta_vec_size), dtype=float) for _ in range(2)]
    w = np.random.rand(1, beta_vec_size)

    done = 0

    output_dict = {"remote_beta": w.tolist(), "computation_phase": "remote_0"}

    cache_dict = {
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "tol": tol,
        "eta": eta,
        "count": count,
        "w": w.tolist(),
        "mt": mt.tolist(),
        "vt": vt.tolist(),
        "iter_flag": done,
        "loss": loss,
        "iterations": iterations
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_output)


def remote_1(args):
    input_list = args["input"]
    beta1 = args["cache"]["beta1"]
    beta2 = args["cache"]["beta2"]
    eps = args["cache"]["eps"]
    tol = args["cache"]["tol"]
    eta = args["cache"]["eta"]
    count = args["cache"]["count"]
    w = np.array(args["cache"]["w"], dtype=float)
    mt = args["cache"]["mt"]
    vt = args["cache"]["vt"]
    loss = args["cache"]["loss"]
    done = args["cache"]["iter_flag"]
    iterations = args["cache"]["iterations"]

    # gather gradients and obj func value for locals
    if len(input_list) == 1:
        grad_remote = [
            np.array(args["input"][site]["local_grad"]) for site in input_list
        ]
        grad_remote = grad_remote[0]

        total_loss = [args["input"][site]["obj_val"] for site in input_list]
        total_loss = total_loss[0]
    else:
        grad_remote = sum([
            np.array(args["input"][site]["local_grad"]) for site in input_list
        ])

        total_loss = sum(
            [np.array(args["input"][site]["obj_val"]) for site in input_list])

    count = count + 1

    loss_diff = abs(loss - total_loss)
    
    if loss_diff <= tol or np.isnan(loss_diff) or count > iterations:
        done = 1

    if done:
        output_dict = {"avg_beta_vector": w.tolist()}

        computation_output = {
            "output": output_dict,
            "success": True,
        }
    else:
        mt = beta1 * np.array(mt) + (1 - beta1) * grad_remote
        vt = beta2 * np.array(vt) + (1 - beta2) * (grad_remote**2)

        m = mt / (1 - beta1**count)
        v = vt / (1 - beta2**count)

        w = w - eta * m / (np.sqrt(v) + eps)

        output_dict = {
            "remote_beta": w.tolist(),
            "computation_phase": "remote_1"
        }

        cache_dict = {
            "count": count,
            "w": w.tolist(),
            "mt": mt.tolist(),
            "vt": vt.tolist(),
            "iter_flag": done
        }

        computation_output = {
            "output": output_dict,
            "cache": cache_dict,
        }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, "computation_phase"))

    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
