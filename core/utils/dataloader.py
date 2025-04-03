
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import ProgressBar
from configs.config import Config

# progress bar
bar = ProgressBar()

def clean_jobs(batch_task_path=Config.batch_task_path,
                cleaned_task_path=Config.cleaned_task_path):
    """
    :return: sampling dependent jobs batch_task.csv 
    """
    if os.path.exists(cleaned_task_path):
        print("Dataset batch_task.csv is already selected.")
        return
    
    if not os.path.exists(batch_task_path):
        print('batch_task.csv is not exist! Please download it from '
              'http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz!')
        return

    print("Cleaning data from batch_task.csv ...")

    columns = ['task_name', 'instance_num', 'job_name', 'task_type', 'status',
                'start_time', 'end_time', 'plan_cpu', 'plan_mem']
    df = pd.read_csv(batch_task_path, header=None, names=columns)

    # only need task_name and job_name (for now)
    df_jobs = df[['task_name', 'job_name']]
    
    # fetch dependent tasks according to column 'task_name'
    booltask = ~df_jobs['task_name'].str.contains('task|MergeTask')
    df_jobs = df_jobs[booltask]

    print(df_jobs.describe)

    df_jobs.to_csv(cleaned_task_path, mode='a', header=True, index=0)

def sample_jobs(cleaned_task_path=Config.cleaned_task_path,
                    sampled_batch_task_path=Config.sampled_batch_task_path):
    if not os.path.exists(cleaned_task_path):
        print('cleaned_batch_task.csv is not exist! Please first run clean_jobs()')
        return
    
    df = pd.read_csv(cleaned_task_path)

    print(df.describe)

    df_len = df.shape[0]
    
    num_sampled = Config.num_selected_job
    DAGs = df.loc[0: 0]

    cnt_job = np.zeros(4)

    idx = 0
    while idx < df_len:
        dag_name = df.loc[idx, 'job_name']
        dag_len = 0
        while (idx + dag_len < df_len) and (df.loc[idx + dag_len, 'job_name'] == dag_name):
            dag_len += 1
        if 2 <= dag_len <= 10:
            if cnt_job[0] < num_sampled[0]:
                DAGs = pd.concat([DAGs, df.loc[idx: idx + dag_len - 1].copy()], axis=0)
                cnt_job[0] += 1
        if 11 <= dag_len <= 50:
            if cnt_job[1] < num_sampled[1]:
                DAGs = pd.concat([DAGs, df.loc[idx: idx + dag_len - 1].copy()], axis=0)
                cnt_job[1] += 1
        elif 51 <= dag_len <= 100:
            if cnt_job[2] < num_sampled[2]:
                DAGs = pd.concat([DAGs, df.loc[idx: idx + dag_len - 1].copy()], axis=0)
                cnt_job[2] += 1
        elif dag_len > 100:
            if cnt_job[3] < num_sampled[3]:
                DAGs = pd.concat([DAGs, df.loc[idx: idx + dag_len - 1].copy()], axis=0)
                cnt_job[3] += 1
        idx = idx + dag_len
        
        percent = idx / float(df_len) * 100
        bar.update(percent)
    
    DAGs.to_csv(sampled_batch_task_path, index=0)

    # Define the values of the five items
    items = list(cnt_job)

    # Create a bar chart to visualize the values
    plt.bar(['2~10', '11~50', '51~100', '>100'], items)

    # Use a logarithmic scale on the y-axis
    plt.yscale('log')

    # Add labels and title to the plot
    plt.xlabel('Jobs range')
    plt.ylabel('Number of tasks (logarithmic scale)')

    # Display the plot
    plt.show()

def get_topological_order(sampled_batch_task_path=Config.sampled_batch_task_path,
                        batch_task_topological_order_path=Config.batch_task_topological_order_path):
    if os.path.exists(batch_task_topological_order_path):
        print('Jobs\' topological order (batch_task_topological_order.csv) has been obtained.')
        return

    if not os.path.exists(sampled_batch_task_path):
        print('The sampling procedure has not been executed! Please sampling jobs firstly.')
        return

    df = pd.read_csv(sampled_batch_task_path)
    rows = df.shape[0]
    idx = 0

    sorted_num = 0

    print('Getting topological order for %d rows ...' % rows)
    while idx < rows:
        # get a DAG
        dag_name = df.loc[idx, 'job_name']
        dag_len = 0
        while (idx + dag_len < rows) and (df.loc[idx + dag_len, 'job_name'] == dag_name):
            dag_len = dag_len + 1
        DAG = df.loc[idx: idx + dag_len].copy()

        # get the number and dependencies of each function of the DAG
        funcs_num = np.zeros(dag_len)
        dependencies = [[] * 1] * dag_len
        for i in range(dag_len):
            name_str_list = DAG.loc[i + idx, 'task_name'].split('_')
            name_str_list_len = len(name_str_list)
            func_str_len = len(name_str_list[0])
            func_num = int(name_str_list[0][1:func_str_len])
            dependent_funcs = []
            for j in range(name_str_list_len):
                if j == 0:
                    continue
                if name_str_list[j].isnumeric():
                    # the func's dependencies
                    dependent_func_num = int(name_str_list[j])
                    dependent_funcs.append(dependent_func_num)
            funcs_num[i] = func_num
            dependencies[i] = dependent_funcs

        # sort the functions accroding to their dependencies
        funcs_left = dag_len
        DAG_sorted = DAG.copy()
        while funcs_left > 0:
            for i in range(len(dependencies)):
                if len(dependencies[i]) == 0:
                    running_func = i
                    dependencies[i].append(-1)
                    break
            func_running = int(funcs_num[running_func])
            for i in range(len(dependencies)):
                if dependencies[i].count(func_running) > 0:
                    dependencies[i].remove(func_running)
            DAG_sorted.loc[dag_len - funcs_left + idx] = DAG.loc[running_func + idx].copy()
            funcs_left = funcs_left - 1
        df.loc[idx: idx + dag_len - 1] = DAG_sorted.copy()
        idx = idx + dag_len
        
        percent = sorted_num / float(Config.num_jobs) * 100
        bar.update(percent)

        sorted_num = sorted_num + 1
        
    df.to_csv(batch_task_topological_order_path, index=False)

if __name__ == '__main__':
    # clean_jobs()
    sample_jobs()
    get_topological_order()
    # pass