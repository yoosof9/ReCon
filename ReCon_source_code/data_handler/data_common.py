import pandas as pd
import numpy as np
from data_handler import data_utils
import torch.utils.data as data


def get_graph(inputgraph, delimiter):
    import networkx as nx
    edges = np.loadtxt(inputgraph, delimiter=delimiter, dtype=int)
    graph = nx.Graph()
    graph.add_edges_from(edges[:, :2])
    return graph


def make_zero_incremental(arr):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    count = len(np.unique(arr))
    min = np.min(arr)
    max = np.max(arr)
    if min == 0 and max == (count-1):
        return arr
    elif min == 1 and max == count:
        return arr - 1
    else:
        unique_values = np.unique(arr)
        sorted_values = np.sort(unique_values)
        for idx, val in enumerate(sorted_values):
            arr[arr == val] = idx
        return arr


def make_zero_incremental_list(arr_list):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    arr = list()
    for ar in arr_list:
        for i in ar:
            arr.append(i)

    unique_values = np.unique(arr)
    sorted_values = np.sort(unique_values)
    for idx, val in enumerate(sorted_values):
        for ar in arr_list:
            ar[ar == val] = idx
    return arr_list


def make_zero_incremental_list_with_mapping(series_list):
    """
    :param arr: a vector
    :return: zero incremental vector
    """
    new_series_list = [pd.Series((ss for ss in s)) for s in series_list]
    arr = pd.concat(new_series_list)

    unique_values = pd.unique(arr)
    # sorted_values = np.sort(unique_values)
    d = dict()
    for idx, val in enumerate(unique_values):
        d[val] = idx
    new_list = list()
    for ar in new_series_list:
        new_list.append(ar.map(d))
    return new_list, d


def get_data(args, config):
    train_data, test_data_classification, test_data_ranking, user_num ,item_num, train_mat, val_data = data_utils.load_all(config=config)
    top_k_list_for_rec_performance = [1, 5, 10, 50, 100]
    top_k_list = [1, 5, 10, 50, 100, 200, 500, 1000]

    train_dataset = data_utils.CustomDataset(
    train_data, user_num, item_num, train_mat, args.num_ng, True)
    train_loader = data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = data_utils.CustomDataset(
    val_data, user_num, item_num, train_mat, args.num_ng, True)
    val_loader = data.DataLoader(val_dataset,
    batch_size=500, shuffle=False, num_workers=4)


    test_dataset_classification = data_utils.CustomDataset(
		test_data_classification, user_num, item_num, train_mat, 0, False)
    test_dataset_ranking = data_utils.CustomDataset(
		test_data_ranking, user_num, item_num, train_mat, 0, False)
    
    test_loader_classification = data.DataLoader(test_dataset_classification,
		batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader_ranking = data.DataLoader(test_dataset_ranking,
		batch_size=args.batch_size, shuffle=False, num_workers=0)
      
    return user_num,item_num,top_k_list_for_rec_performance,top_k_list,train_loader,test_loader_classification, test_loader_ranking, val_loader
