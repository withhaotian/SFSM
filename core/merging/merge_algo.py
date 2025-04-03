import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from collections import deque

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.figure import DAGFigure
from utils.tools import find_key_by_value
from configs.config import Config

random.seed(Config.seed)
np.random.seed(Config.seed)

class MergeAlgorithm(object):
    """
        Load-Aware Function Merging Implementation, including vertical merging and horizontal merging
    """
    def __init__(self, config) -> None:
        self._config = config

    def get_dag_with_topology(self, job_dag):
        '''
            get topology sort of one DAG job
        '''
        return list(nx.topological_sort(job_dag))
    
    def merging_and_inherit_vertical(self, job_dag, node_a, node_b):
        '''
            merge node A and node B via vertical approach
        '''
        # merge computations
        attrs = {node_b: {'computations': job_dag.nodes[node_a]['computations'] + job_dag.nodes[node_b]['computations']}}
        nx.set_node_attributes(job_dag, attrs)

        # merge data
        predecessors_of_a = list(job_dag.predecessors(node_a))
        for predeccessor in predecessors_of_a:
            job_dag.add_edge(predeccessor, node_b, data=job_dag[predeccessor][node_a]['data'])
        
        job_dag.remove_node(node_a)
        return job_dag

    def merging_and_inherit_horizontal(self, job_dag, node_a, node_b):
        '''
            merge node A and node B via horizontal approach
        '''

        # print(node_a, '     -----      ', node_b)
        # print(nx.get_node_attributes(job_dag, 'parallel'))

        # merge computations
        # attrs = {node_a: {'computations': max(job_dag.nodes[node_a]['computations'], job_dag.nodes[node_b]['computations']), 
        #                 'parallel': job_dag.nodes[node_a]['parallel'] + [job_dag.nodes[node_b]['computations']]}}
        attrs = {node_a: {'parallel': job_dag.nodes[node_a]['parallel'] + [job_dag.nodes[node_b]['computations']]}}
        nx.set_node_attributes(job_dag, attrs)

        # merge data
        successors_of_b = set(job_dag.successors(node_b))
        successors_of_a = set(job_dag.successors(node_a))
        
        # note that if node a and node b has the same successor node c, has to merge the sum of transmission data (implement by set)
        common_successors = successors_of_b.intersection(successors_of_a)
        for successor in common_successors:
            job_dag[node_a][successor]['data'] = max(job_dag[node_a][successor]['data'], job_dag[node_b][successor]['data'])

        for successor in successors_of_b:
            if successor not in common_successors:
                job_dag.add_edge(node_a, successor, data=job_dag[node_b][successor]['data'])

        predecessors_of_b = set(job_dag.predecessors(node_b))
        predecessors_of_a = set(job_dag.predecessors(node_a))

        common_predecessors = predecessors_of_b.intersection(predecessors_of_a)
        for predecessor in common_predecessors:
            if job_dag.nodes[predecessor]['is_merged'] is True:
                job_dag[predecessor][node_a]['data'] = max(job_dag[predecessor][node_a]['data'], job_dag[predecessor][node_b]['data'])  # shared transmission
        
        for predecessor in predecessors_of_b:
            if predecessor not in common_predecessors:
                job_dag.add_edge(predecessor, node_a, data=job_dag[predecessor][node_b]['data'])

        # is_merged flag
        attrs = {node_a: {'is_merged': True}}
        nx.set_node_attributes(job_dag, attrs)

        job_dag.remove_node(node_b)

        return job_dag

    def vertical_merging(self, job_dag):
        '''
            vetical merging of functions with node with one successor and its successor has one predecessor
        '''

        entry_node = 0
        for node in job_dag.nodes():
            if job_dag.out_degree(node) == 0:
                exit_node = node
    
        _bfs = nx.bfs_tree(job_dag, entry_node)
        for node in _bfs.nodes():
            # print(node)
            # find the node A only has one successor and A's successor B also has only one predeccor, merge A and B
            if node in job_dag.nodes:
                successors = list(job_dag.successors(node))
                if (node != entry_node) and (exit_node not in successors) and len(successors) == 1:
                    successor = successors[0]
                    if len(list(job_dag.predecessors(successor))) == 1 and list(job_dag.predecessors(successor))[0] == node:
                        job_dag = self.merging_and_inherit_vertical(job_dag, node, successor)

        return job_dag
    
    def horizontal_merging(self, job_dag):
        '''
            load-aware horizontal merging of functions with depth-level
        '''

        entry_node = 0
        for node in job_dag.nodes():
            if job_dag.out_degree(node) == 0:
                exit_node = node
        
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
        
        for key, val in node_levels.items():
            
            for node in val:
                if node in job_dag.nodes:
                    # find successors with different level, find more than 2 successors
                    if len(list(job_dag.successors(node))) >= 2:
                        successors_by_level = {}
                        for successor in job_dag.successors(node):
                            level = find_key_by_value(node_levels, successor)
                            if successors_by_level.get(level) is None:
                                successors_by_level[level] = [successor]
                            else:
                                successors_by_level[level].append(successor)
                        
                        for levels, successors in successors_by_level.items():
                            if len(successors) > 1:
                                # print('successors,', successors)
                                # sort by computations
                                dict_node_comp = {}
                                for successor in successors:
                                    dict_node_comp[successor] = job_dag.nodes[successor]['computations']
                                
                                sorted_by_computations = sorted(dict_node_comp.items(),key = lambda x:x[1], reverse=True)

                                i = 0
                                j = len(sorted_by_computations) - 1
                                
                                while i < j:
                                    nodes_to_merge = []

                                    while i < j and i < self._config.parallel_funcs // 2:
                                        nodes_to_merge.append(sorted_by_computations[i][0])
                                        nodes_to_merge.append(sorted_by_computations[j][0])
                                        i += 1
                                        j -= 1
                                    
                                    if i < j and self._config.parallel_funcs % 2 != 0:
                                        nodes_to_merge.append(sorted_by_computations[i][0])
                                        i += 1 
                                    
                                    # print('=== node to merge ===', nodes_to_merge)

                                    # n nodes merge n-1 times
                                    for k in range(len(nodes_to_merge) - 1):
                                        job_dag = self.merging_and_inherit_horizontal(job_dag, nodes_to_merge[0], nodes_to_merge[k+1])

            # # sort by computations
            # dict_node_comp = {}
            # for node in val:
            #     dict_node_comp[node] = job_dag.nodes[node]['computations']
            #     # print(dict_node_comp)

            # sorted_by_computations = sorted(dict_node_comp.items(),key = lambda x:x[1], reverse=True)
            # # print('######################')
            # # print(sorted_by_computations)
            
            # if len(sorted_by_computations) < 2:
            #     continue
            
            # i = 0
            # j = len(sorted_by_computations) - 1
            
            # while i < j:
            #     nodes_to_merge = []

            #     while i < j and i < self._config.parallel_funcs // 2:
            #         nodes_to_merge.append(sorted_by_computations[i][0])
            #         nodes_to_merge.append(sorted_by_computations[j][0])
            #         i += 1
            #         j -= 1
                
            #     if i < j and self._config.parallel_funcs % 2 != 0:
            #         nodes_to_merge.append(sorted_by_computations[i][0])
            #         i += 1 
                
            #     # print('=== node to merge ===', nodes_to_merge)

            #     # n nodes merge n-1 times
            #     for k in range(len(nodes_to_merge) - 1):
            #         job_dag = self.merging_and_inherit_horizontal(job_dag, nodes_to_merge[0], nodes_to_merge[k+1])
        
        return job_dag

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_node(0, computations=4, is_merged=False)
    G.add_node(1, computations=8, is_merged=False)
    G.add_node(2, computations=12, is_merged=False)
    G.add_node(3, computations=7, is_merged=False)
    G.add_node(4, computations=14, is_merged=False)
    G.add_node(5, computations=9, is_merged=False)
    G.add_edge(0, 1, data=9)
    G.add_edge(0, 2, data=9)
    G.add_edge(0, 3, data=9)
    G.add_edge(1, 4, data=17)
    G.add_edge(2, 4, data=10)
    G.add_edge(3, 4, data=4)
    G.add_edge(3, 5, data=5)

    dag_fig = DAGFigure()
    dag_fig.visual(G, 'G_job', True)

    merge_algo = MergeAlgorithm()
    _G = merge_algo.horizontal_merging(G)

    print('************* After Horizontal Merging ****************')
    dag_fig.visual(_G, '_G_job', True)