# -*- coding: utf-8 -*-
# author: Jakub Skałecki (jakub.skalecki@gmail.com)
import random


def sample_with_replacement(data, sample_size, replacement=True):
    if replacement:
        return [random.choice(data) for _ in xrange(sample_size)]
    else:
        return random.sample(data, sample_size)