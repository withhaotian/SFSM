from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

@dataclass
class Config:
    
    """ number of edge servers"""
    num_es : int

    """ number of users"""
    num_users : int

    """ max number of connections in edge networks (i.e., lower than # edge servers)"""
    num_max_connections = 7

    """ limit of bandwidth"""
    band_lower, band_upper = 20, 800

    """ limit of processing power"""
    proc_lower, proc_upper = 50, 1000

    """ Execution Workload Mode:  (1) data-intensive / (2) computation-intensive / (3) data and computation-intensive"""
    job_mode : int

    """ 
        limit of data transmation from one function to another
        normal: [5, 200]
        data-intensive: [100, 3000]
    """
    data_lower : int
    data_upper : int

    """ 
        limit of required computation resources
        normal: [10, 300]
        computation-intensive: [200, 5000]
    """
    comp_lower : int
    comp_upper : int

    """ limit of processing power of local devices (only used for comparisons)"""
    local_proc_lower, local_proc_upper = 40, 800

    """ limit of bandwidth of local devices (only used for comparisons)"""
    local_band_lower, local_band_upper = 20, 2000

    """ transmission rate between edge to cloud"""
    band_edge2cloud_lower, band_edge2cloud_upper = 20, 1000

    """ storage capacity for configuration deployment"""
    cap_conf = 300

    """ Dataset specification"""
    home_addr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    batch_task_path = os.path.join(home_addr, 'datasets/batch_task.csv')
    cleaned_task_path = os.path.join(home_addr, 'datasets/cleaned_batch_task.csv')
    sampled_batch_task_path = os.path.join(home_addr, 'datasets/sampled_batch_task.csv')
    batch_task_topological_order_path = os.path.join(home_addr, 'datasets/batch_task_topological_order.csv')

    """
    |Number of applications(jobs) are|: 3175025
    |2   ~  10   jobs|: 1840085
    |11  ~  50   jobs|: 213130
    |51  ~  100  jobs|: 1965
    |101 ~  150  jobs|: 115
    |    >  150  jobs|: 4
    sampled distributions of applications (i.e., jobs)
    # level 1: 2   ~  10   jobs (8381)      
    # level 2: 11  ~  50   jobs (10000)     
    # level 3: 51  ~  100  jobs (1500)      
    # level 4: > 100       jobs (only 119)                  total: 20k applications
    # maximum number of functions is 203
    """
    num_selected_job = [8381, 10000, 1500, 119]
    num_jobs = 20000

    """ random seed """
    seed = 42

    """ computing parallel capacity for edge servers"""
    parallel_funcs = 3

    """ learning times for bandit agent"""
    learning_times = 1000

    """ episodes for DRL agent"""
    episodes = 1000

    """ time slots for Lyapunov optimization"""
    time_slots = 1

    """ task arrival rate """
    rho : float

    """ control paprameter V"""
    V : float

    """ computation cost factor"""
    comp_cost = 0.8

    """ transmission cost factor"""
    trans_cost = 0.8

    """ execution cost threshold"""
    c_threshold : int


if __name__ == '__main__':
    print(Config.home_addr)
    print(Config.selected_batch_task_path)