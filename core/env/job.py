import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.figure import DAGFigure
from utils.tools import str_to_int
from configs.config import Config

random.seed(Config.seed)
np.random.seed(Config.seed)

class Job(object):
    """
        DAG job (a.k.a., application)
    """
    def __init__(self, job_id, proc_time, trans_time, num_tasks, deadline):
        self.job_id = job_id                # job id,
        self.proc_time = proc_time          # processing time
        self.trans_time = trans_time        # transmission time bewteen tasks form A to B
        self.num_tasks = num_tasks          # number of tasks (or functions)

        self.deadline = deadline            # processing deadline

    @staticmethod
    def generate_dag_from_alibaba_trace_data(config, job: DataFrame, job_name=None):
        """
        initialize a DAG from one request(job)
        """
        # print(job.columns)
        task_name_list = job.loc[:, 'task_name']

        G = nx.DiGraph()
        for task_name in task_name_list:
            dependencies = task_name.split('_')
            dependencies[0] = dependencies[0][1:]
        
            # if not dependencies[0].isdigit():
            #     G.add_node(task_name)
            #     continue

            curr_task = str_to_int(dependencies[0])
            if len(dependencies) == 1:
                G.add_node(curr_task, computations=random.randint(config.comp_lower, config.comp_upper), is_merged=False)
            else:
                i = 1
                while i < len(dependencies):
                    if dependencies[i].isdigit():   # only for number
                        dependency = str_to_int(dependencies[i])
                        G.add_edge(dependency, curr_task, data=random.randint(config.data_lower, config.data_upper))
                        computation = random.randint(config.comp_lower, config.comp_upper)
                        attrs = {dependency: {'computations': computation, 'is_merged': False, 'parallel': [computation]}}
                        nx.set_node_attributes(G, attrs)
                    i += 1
        
        # refine the data transmission of one node to its successors
        for node in G.nodes():
            if G.out_degree(node) > 1:
                successors = G.successors(node)
                data_size = random.randint(config.data_lower, config.data_upper)
                for successor in successors:
                    G[node][successor]['data'] = data_size

        # Add a dummy entry node
        entry_node = 0
        G.add_node(entry_node, computations=0, is_merged=False, parallel=[0])

        # Connect dummy entry node to all nodes in G that have no incoming edges
        for node in G.nodes():
            if G.in_degree(node) == 0 and entry_node != node:
                G.add_edge(entry_node, node, data=0)

        # Add a dummy exit node
        exit_node = len(task_name_list) + 1
        G.add_node(exit_node, computations=0, is_merged=False, parallel=[0])

        # Connect all nodes in G that have no outgoing edges to the dummy exit node
        for node in G.nodes():
            if G.out_degree(node) == 0 and exit_node != node:
                G.add_edge(node, exit_node, data=0)
                computation = random.randint(config.comp_lower, config.comp_upper)
                attrs = {node: {'computations': computation, 'is_merged': False, 'parallel': [computation]}}
                nx.set_node_attributes(G, attrs)

        if job_name is None:
            job_name = job['job_name'].loc[job.index[0]]
        
        # not DAG
        if not nx.is_directed_acyclic_graph(G):
            return G, 'error'

        return G, job_name


    @staticmethod
    def renumber_dag(job_dag):
        """
        renumber the DAG with serial number
        """

        new_nodes_mapping = {}

        # No. [0, 1, ..., new_num_funcs - 1]
        new_num_funcs = job_dag.number_of_nodes()

        entry_node = 0
        exit_node = new_num_funcs - 1

        i = 1

        for node in job_dag.nodes():
            if node == entry_node:
                continue
            # exit_node
            if job_dag.out_degree(node) == 0:
                new_nodes_mapping[node] = exit_node
            else:
                new_nodes_mapping[node] = i
                i += 1
        
        # print(new_nodes_mapping)

        new_job_dag = nx.relabel_nodes(job_dag, new_nodes_mapping)

        return new_job_dag

if __name__ == '__main__':
    G = nx.DiGraph()

    G.add_nodes_from([0, 3, 5, 6, 9])
    G.add_edges_from([(0, 3), (3, 5), (5, 6), (6, 9)])

    dag_fig = DAGFigure()
    dag_fig.visual(G, '')

    _G = Job.renumber_dag(G)
    dag_fig.visual(_G, '')