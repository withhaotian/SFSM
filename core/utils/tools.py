import pprint
import random
import numpy as np
from pandas import DataFrame
import sys
import os
import networkx as nx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config

random.seed(Config.seed)

def print_edge_networks(G, W):
    print('\nThe connected graph of edge servers (represented by adjcent matrix):')
    pprint.pprint(G)
    print('\n====> throughput of each link <====')
    pprint.pprint(W)

def get_one_job(jobs : DataFrame, start_idx=0):
    """
    get one job from alibaba batch_task datasets
    """

    job_name = jobs.loc[start_idx, 'job_name']
    rows = jobs.shape[0]
    task_nums = 0
    while (start_idx + task_nums < rows) and (jobs.loc[start_idx + task_nums, 'job_name'] == job_name):
        task_nums = task_nums + 1
    
    job = jobs.loc[start_idx: start_idx + task_nums - 1].copy()
    
    next_idx = start_idx + task_nums
    
    return job, next_idx

def str_to_int(str):
    try:
        return int(str)
    except ValueError:
        return None

def init_computation(length):
    '''
        initialize the required computatuion resources for one DAG
    '''
    
    res = np.zeros(length)

    for i in range(1, length-1):
        res[i] = random.randint(Config.comp_lower, Config.comp_upper)
    
    return res

def find_key_by_value(dic, value):
    for key, val in dic.items():
        for item in val:
            if item == value:
                return key
    return None

def get_node_level(job_dag):
    # get level based on topological order
    topological_order = list(nx.topological_sort(job_dag))
    
    node_levels = {}
    for node in topological_order:
        predecessors = list(job_dag.predecessors(node))
        if len(predecessors) == 0:
            if node_levels.get(0) is None:
                node_levels[0] = [node]
            else:
                node_levels[0].append(node)
        else:
            max_level = 0
            for predecessor in job_dag.predecessors(node):
                # print('node levels,', node_levels)
                # print('predeccessor,', predecessor)
                max_level = max(max_level, find_key_by_value(node_levels, predecessor) + 1)
            if node_levels.get(max_level) is None:
                node_levels[max_level] = [node]
            else:
                node_levels[max_level].append(node)

    # print(node_levels)
    return node_levels

def get_path_cost(job_dag):
    # Calculate b_level
    max_combined_sum = {node: 0 for node in job_dag.nodes}

    for node in reversed(list(nx.topological_sort(job_dag))):
        if node == job_dag.number_of_nodes() - 1:
            max_combined_sum[node] += job_dag.nodes[node]['computations']
            continue
        for succ in job_dag.successors(node):
            edge_weight = job_dag[node][succ]['data']
            combined_sum = job_dag.nodes[node]['computations'] + edge_weight + max_combined_sum[succ]
            max_combined_sum[node] = max(max_combined_sum[node], combined_sum)

    return max_combined_sum

def get_function_cost(job_dag):
    max_node_sum = {node: 0 for node in job_dag.nodes}

    for node in reversed(list(nx.topological_sort(job_dag))):
        if node == job_dag.number_of_nodes() - 1:
            max_node_sum[node] = job_dag.nodes[node]['computations']
            continue
        for succ in job_dag.successors(node):
            max_node_sum[node] = max(max_node_sum[node], job_dag.nodes[node]['computations'] + max_node_sum[succ])
    
    return max_node_sum