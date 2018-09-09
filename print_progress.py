# -*- coding: utf-8 -*-

import sys
import time

def print_progress(iteration, total_iterations, update_interval_sec = 1.0):
    global _time_started, _time_updated
    
    t = time.time()
    i = iteration
    if i==0:
        _time_started = _time_updated = t
    elif ((t - _time_updated) > update_interval_sec) or i == (total_iterations - 1) :
        _time_updated = t
        t_elapsed = t - _time_started    
        i_ = i + 1
        progress = i_ / total_iterations
        t_est_total = t_elapsed / progress
        t_est_remained = t_est_total - t_elapsed
        sys.stdout.write('\r Progress:{0: 5.2f} % '.format(100 * i_ / total_iterations) \
                         + ' | Processed:{0: 7d}/{1: 6d}'.format(i_, (total_iterations))\
                         + ' | Elapsed: {0: 6.0f} sec'.format(t_elapsed)\
                         + ' | Est total: {0: 8.0f} sec'.format(t_est_total)\
                         + ' | Est remained: {0: 8.0f} sec'.format(t_est_remained)\
                        )
        sys.stdout.flush()
    if i == (total_iterations - 1):
        print()