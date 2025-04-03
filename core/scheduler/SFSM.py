import networkx as nx
import numpy as np

import sys
import os
from configs.config import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import find_key_by_value, get_node_level, get_path_cost, get_function_cost

np.random.seed(Config.seed)

def fill_theta_context(job_dag, num_functions, theta_context_dim):

    x_theta = np.zeros((theta_context_dim, num_functions))

    node_levels = get_node_level(job_dag)

    node_path_cost = get_path_cost(job_dag)

    node_cost = get_function_cost(job_dag)

    # theta context, dim = 7
    # [current id, current level, node cost, node&path cost, computation cost, sum of data from pred, sum of data from succ]
    for i in range(num_functions):
        x_theta[0][i] = i
        x_theta[1][i] = find_key_by_value(node_levels, i)

        x_theta[2][i] = node_cost[i]
        x_theta[3][i] = node_path_cost[i]

        x_theta[4][i] = job_dag.nodes[i]['computations']

        pre_data = 0
        for pre in job_dag.predecessors(i):
            pre_data += job_dag[pre][i]['data']
        x_theta[5][i] = pre_data

        succ_data = 0
        for succ in job_dag.successors(i):
            succ_data += job_dag[i][succ]['data']
        x_theta[6][i] = succ_data
    return x_theta

def get_context(x_theta, num_functions):
    listC_x = []
    for i in range(num_functions):
        temp = np.sqrt(np.matmul(x_theta[:, [i]].T, x_theta[:, [i]]))
        listC_x.append(temp[0][0])
    Cx = pow(max(listC_x), 2)
    return Cx

class SFSM():
    def __init__(self, config, job_dag, G_edge, W_edge, es_all):
        self._config = config
        self.job_dag = job_dag      # DAG of functions

        self.G_edge = G_edge        # connection metrix of edge servers
        self.W_edge = W_edge        # transmission rate of edge servers
        self.es_all = es_all        # list of edge servers

        self.num_edge_servers = len(self.es_all)
        self.num_functions = self.job_dag.number_of_nodes()

        self.theta_context_dim = 7
        self.x_theta = fill_theta_context(self.job_dag, self.num_functions, self.theta_context_dim)
        self.C_x = get_context(self.x_theta, self.num_functions)

        self.delta = 0.1
        self.C_noise = 0.1
        self.C_theta = 20
        # print('c_noise ', self.C_noise, 'c_theta', self.C_theta)
        self.A = np.array([np.diag(np.random.randint(1, 9, size=self.theta_context_dim)) for _ in range(self.num_edge_servers)])
        self.b = np.array([np.zeros((self.theta_context_dim, 1)) for _ in range(self.num_edge_servers)])
        self.alpha = self.C_theta + np.sqrt(np.log((1 + self.num_functions * self.num_edge_servers * self.C_x * self.C_x)/self.delta) * self.theta_context_dim)*self.C_noise
        # self.alpha = 0.1

    def get_estimated_super_arm(self):
        all_estimate_action = [_ for _ in range(self.num_functions)]
        all_estimate_delay = [[] for _ in range(self.num_functions)]
        _estimate_makespan = 0.
        
        nodes_order = list(nx.topological_sort(self.job_dag))

        for function in nodes_order:
            estimate_delay = [0 for _ in range(self.num_edge_servers)]
            # print(estimate_delay)
            if function == 0:
                all_estimate_action[0] = 0  # dummy entry node
                continue
            
            # if function == self.num_functions - 1:
            #     all_estimate_action[function] = 0
            #     for pre in self.job_dag.predecessors(function):
            #         min_val = min(all_estimate_delay[pre])
            #         # print('min_val', min_val)
            #         _estimate_makespan = max(_estimate_makespan, min_val)
            #         # print('_estimate_makespan', _estimate_makespan)
            #     continue

            for edge_server in range(self.num_edge_servers):
                A_inv = np.linalg.inv(self.A[edge_server])
                theta = np.matmul(A_inv, self.b[edge_server])

                x_1 = np.copy(self.x_theta[:, [function]])
                x_2 = np.copy(self.x_theta[:, [function]])

                empirical = np.matmul(x_1.T, theta)
                exploration = self.alpha * np.sqrt(np.matmul(np.matmul(x_1.T, A_inv), x_2))

                makespan_pre = 0
                for pre in self.job_dag.predecessors(function):
                    if len(all_estimate_delay[pre]) == 0:
                        continue
                    makespan_pre = max(makespan_pre, min(all_estimate_delay[pre]))

                estimate_delay[edge_server] = (empirical - exploration)[0][0]

            estimate_action = estimate_delay.index(min(estimate_delay))
            all_estimate_action[function] = estimate_action
            all_estimate_delay[function] = estimate_delay

        return all_estimate_action, min(all_estimate_delay[self.num_functions - 1])

    def update_A_b(self, selected_edge_servers, all_estimate_action, actual_makespan):
        for i in selected_edge_servers:
            temp_1 = np.zeros((self.theta_context_dim, self.theta_context_dim))
            temp_2 = np.zeros((self.theta_context_dim, 1))
            cnt = 0

            for j in range(len(all_estimate_action)):
                if all_estimate_action[j] == i:
                    temp_1 = temp_1 + np.matmul(self.x_theta[:, [j]], self.x_theta[:, [j]].T)
                    temp_2 = temp_2 + self.x_theta[:, [j]]
                    cnt += 1.0

            self.A[i] = self.A[i] + temp_1
            self.b[i] = self.b[i] + temp_2 * actual_makespan
    
    def get_makespan(self, all_estimate_action):
        nodes_order = list(nx.topological_sort(self.job_dag))
        # print(nodes_order)

        makespan = 0.0

        all_makespan = np.zeros(self.num_functions)

        dummy_entry = 0
        dummy_exit = self.num_functions - 1

        start_time = np.zeros(self.num_functions)
        server_runtime = np.zeros(self.num_edge_servers)
        
        for function in nodes_order:
            if function == dummy_entry:
                continue

            # compute the makespan
            if function == dummy_exit:
                last_finised = 0
                for pre in self.job_dag.predecessors(function):
                    if all_makespan[pre] > last_finised:
                        last_finised = all_makespan[pre]
                makespan = last_finised
                # print(makespan)
                break
            
            # first execution except dummy entry
            if dummy_entry in self.job_dag.predecessors(function):
                
                all_makespan[function] = max(self.job_dag.nodes[function]['parallel']) * 1.0 / \
                    self.es_all[all_estimate_action[function]].proc_power + server_runtime[all_estimate_action[function]]
                
                server_runtime[all_estimate_action[function]] = all_makespan[function]
                start_time[function] = server_runtime[all_estimate_action[function]] - \
                    max(self.job_dag.nodes[function]['parallel']) * 1.0 / self.es_all[all_estimate_action[function]].proc_power
                continue
            
            min_process_begin_time = 0
            for pre in self.job_dag.predecessors(function):
                min_exec = float('inf')
                where_deployed_pre = all_estimate_action[pre]
                if where_deployed_pre == all_estimate_action[function]:
                    trans_cost = 0
                else:
                    trans_cost = self.job_dag[pre][function]['data'] * 1.0 / self.W_edge[where_deployed_pre][all_estimate_action[function]]
                
                tmp = all_makespan[pre] + trans_cost
                if tmp > min_process_begin_time:
                    min_process_begin_time = tmp

            if min_process_begin_time > server_runtime[all_estimate_action[function]]:
                process_begin_time = min_process_begin_time
            else:
                process_begin_time = server_runtime[all_estimate_action[function]]
            
            all_makespan[function] = max(self.job_dag.nodes[function]['parallel']) * 1.0 / \
                self.es_all[all_estimate_action[function]].proc_power + process_begin_time
            
            server_runtime[all_estimate_action[function]] = all_makespan[function]
            start_time[function] = server_runtime[all_estimate_action[function]] - \
                    max(self.job_dag.nodes[function]['parallel']) * 1.0 / self.es_all[all_estimate_action[function]].proc_power

        return makespan

    def get_objective_value(self, all_estimate_action, Q, V, c_threshold):

        nodes_order = list(nx.topological_sort(self.job_dag))
        # print(nodes_order)

        makespan = 0.0
        cost = 0.0

        all_makespan = np.zeros(self.num_functions)

        dummy_entry = 0
        dummy_exit = self.num_functions - 1

        start_time = np.zeros(self.num_functions)
        server_runtime = np.zeros(self.num_edge_servers)
        
        for function in nodes_order:
            if function == dummy_entry:
                continue

            # compute the makespan
            if function == dummy_exit:
                last_finised = 0
                for pre in self.job_dag.predecessors(function):
                    if all_makespan[pre] > last_finised:
                        last_finised = all_makespan[pre]
                makespan = last_finised
                # print(makespan)
                break
            
            # first execution except dummy entry
            if dummy_entry in self.job_dag.predecessors(function):
                
                all_makespan[function] = max(self.job_dag.nodes[function]['parallel']) * 1.0 / \
                    self.es_all[all_estimate_action[function]].proc_power + server_runtime[all_estimate_action[function]]
                
                for i in range(len(self.job_dag.nodes[function]['parallel'])):
                    cost += (self.job_dag.nodes[function]['parallel'][i] / self.es_all[all_estimate_action[function]].proc_power) * self._config.comp_cost
                
                server_runtime[all_estimate_action[function]] = all_makespan[function]
                start_time[function] = server_runtime[all_estimate_action[function]] - \
                    max(self.job_dag.nodes[function]['parallel']) * 1.0 / self.es_all[all_estimate_action[function]].proc_power
                continue
            
            min_process_begin_time = 0
            for pre in self.job_dag.predecessors(function):
                min_exec = float('inf')
                where_deployed_pre = all_estimate_action[pre]
                if where_deployed_pre == all_estimate_action[function]:
                    trans_cost = 0
                else:
                    trans_cost = self.job_dag[pre][function]['data'] * 1.0 / self.W_edge[where_deployed_pre][all_estimate_action[function]]

                    cost += trans_cost * self._config.trans_cost
                
                tmp = all_makespan[pre] + trans_cost
                if tmp > min_process_begin_time:
                    min_process_begin_time = tmp

            if min_process_begin_time > server_runtime[all_estimate_action[function]]:
                process_begin_time = min_process_begin_time
            else:
                process_begin_time = server_runtime[all_estimate_action[function]]
            
            all_makespan[function] = max(self.job_dag.nodes[function]['parallel']) * 1.0 / \
                self.es_all[all_estimate_action[function]].proc_power + process_begin_time
            
            for i in range(len(self.job_dag.nodes[function]['parallel'])):
                cost += (self.job_dag.nodes[function]['parallel'][i] / self.es_all[all_estimate_action[function]].proc_power) * self._config.comp_cost
            
            server_runtime[all_estimate_action[function]] = all_makespan[function]
            start_time[function] = server_runtime[all_estimate_action[function]] - \
                    max(self.job_dag.nodes[function]['parallel']) * 1.0 / self.es_all[all_estimate_action[function]].proc_power

        obj_value = V * makespan + Q * (cost - c_threshold)

        return obj_value, makespan, cost
