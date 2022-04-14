#!/usr/bin/env python
# encoding: utf-8

import torch


def euclidean(x1, x2):
    return ((x1-x2)**2).sum().sqrt()


def kth_moment(output_s, output_t, k):
    output_s = (output_s**k).mean(0)
    output_t = (output_t**k).mean(0)
    return euclidean(output_s, output_t)


def moment_distance(output_s, output_t, k):
    loss = 0.
    for i in range(k):
        loss = loss + kth_moment(output_s, output_t, i)
    return loss

