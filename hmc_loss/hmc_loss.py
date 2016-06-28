#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-21
import numpy as np
from functools import reduce
from queue import Queue

def hmc_loss(true_matrix, pred_matrix, graph, root, label_list, cost_list, alpha=None, beta=None, average="micro"):
    label_list = list(label_list)
    cost_list = np.array(cost_list)
    validate_root(graph, root)
    validate_coefficient(alpha, beta)
    validate_average(average)
    validate_list(label_list, cost_list)
    loss = get_loss(true_matrix, pred_matrix, graph, label_list, cost_list, alpha, beta, average)
    return loss

def get_loss(true_matrix, pred_matrix, graph, label_list, cost_list, alpha, beta, average):
    c_matrix = remove_matrix_redunduncy(true_matrix-pred_matrix, label_list, graph)

    if average == "macro":
        if alpha is None or beta is None:
            gamma = get_gamma(true_matrix, average)
            alpha, beta = get_alpha_beta(gamma)
        fn_ci = np.where(c_matrix==1, np.c_[alpha]*cost_list, 0)
        fp_ci = np.where(c_matrix==-1, np.c_[beta]*cost_list, 0)
        loss_list = np.sum(fn_ci+fp_ci, axis=1)
        loss = np.mean(loss_list)
    elif average == "micro":
        if alpha is None or beta is None:
            gamma = get_gamma(true_matrix, average)
            alpha, beta = get_alpha_beta(gamma)
        fn_ci = np.mean(np.where(c_matrix==1, cost_list, 0), axis=0)
        fp_ci = np.mean(np.where(c_matrix==-1, cost_list, 0), axis=0)
        loss = alpha * np.sum(fn_ci) + beta * np.sum(fp_ci)
    return loss

def remove_matrix_redunduncy(matrix, label_list, graph):
    """
        If parent node will be penalized, the node's penalty is removed
    """
    parent_index = get_parent_index_list(graph, label_list)

    m = np.empty(matrix.shape, dtype=int)
    for i, p in enumerate(parent_index):
        if p == []:
            v = matrix[:, i]
        else:
            v = np.where(np.any(matrix[:, p], axis=1)==False, matrix[:,i], 0)
        m[:, i] = v
    return m

def get_gamma(true_matrix, average):
    if average == "macro":
        n_one = np.sum(true_matrix, axis=1)
        n_zero = true_matrix.shape[1] - n_one
        gamma = n_zero / n_one
    elif average == "micro":
        n_one = np.count_nonzero(true_matrix)
        n_zero = reduce(lambda x,y:x*y, true_matrix.shape) - n_one
        gamma = n_zero / n_one
    return gamma

def get_alpha_beta(gamma):
    beta = 2 / (1 + gamma)
    alpha = 2 - beta
    return (alpha, beta)

def get_parent_index_list(graph, label_list):
    """
        Return parent index in label_list.
        To get adaptation for more than one parent, result list is nested.
    """
    parent_index = []
    for label in label_list:
        tmp = []
        for parent in graph.successors(label):
            tmp.append(label_list.index(parent))
        tmp = sorted(tmp)
        parent_index.append(tmp)
    return parent_index

def validate_root(graph, root):
    if len(graph.successors(root)) !=0:
        raise ValueError("Graph direction is wrong.",
            "This function requires bottom-up direction.")
    return 0

def validate_list(label_list, cost_list):
    if len(label_list) != len(cost_list):
        raise ValueError("label_list length doesn't match length of cost_list")
    return 0

def validate_coefficient(alpha, beta):
    if (alpha is None and beta is not None):
        raise ValueError("Beta is None in spite of alpha is used")
    elif (alpha is not None and beta is None):
        raise ValueError("Alpha is None in spite of beta is used")
    return 0

def validate_average(average):
    valid_average = ["micro", "macro"]
    if average not in valid_average:
        raise ValueError("Invalid input of average:{0}".format(average))
    return 0

def get_node_cost(graph, node, cost_dict):
    """
        get cost of input node along
        If parent node cost is not calculated, return 0.
        This node calculate again in the loop by DAG structure.
    """
    cost = 0
    ancestors = graph.successors(node)
    if len(ancestors) == 0:
        return 1
    for ancestor in ancestors:
        try:
            cost += cost_dict[ancestor]/len(graph.predecessors(ancestor))
        except KeyError:
            return 0
    return cost

def get_cost_dict(graph, root):
    """
        Return dictionary that has cost of each node in the graph
    """
    cost_dict = {}
    q = Queue()
    cost_dict[root] = get_node_cost(graph, root, cost_dict)
    for predecessor in graph.predecessors(root):
        q.put(predecessor)
    while q.empty() == False:
        child = q.get()
        cost_dict[child] = get_node_cost(graph, child, cost_dict)
        child_predecessors = graph.predecessors(child)
        for predecessor in child_predecessors:
            q.put(predecessor)
    return cost_dict

def get_cost_list(graph, root, label_list):
    cost_dict = get_cost_dict(graph, root)
    cost_list = [cost_dict[node] for node in label_list]
    return cost_list

def main():
    pass

if __name__ =='__main__':
    main()
