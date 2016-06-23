#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-21
import numpy as np
from queue import Queue

def hmc_loss(true_matrix, pred_matrix, graph, root, label_list, cost_list, alpha=1, beta=1):
    validate_root(graph, root)
    label_list = list(label_list)
    cost_list = np.array(cost_list)
    validate_list(graph, label_list, cost_list)
    loss = get_loss(true_matrix, pred_matrix, graph, label_list, cost_list, alpha, beta)
    return loss

def get_loss(true_matrix, pred_matrix, graph, label_list, cost_list, alpha, beta):
    c_matrix = remove_matrix_redunduncy(true_matrix-pred_matrix, label_list, graph)

    fn_ci = np.where(c_matrix==1, cost_list, 0)
    fp_ci = np.where(c_matrix==-1, cost_list, 0)
    loss_list = alpha * np.sum(fn_ci, axis=1) + beta * np.sum(fp_ci, axis=1)
    loss = np.mean(loss_list)
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

def validate_list(graph, label_list, cost_list):
    if len(graph.nodes()) != len(label_list) :
        raise ValueError("Number of nodes in graph doesn't match length of label_list")
    if len(graph.nodes()) != len(cost_list):
        raise ValueError("Number of nodes in graph doesn't match length of cost_list")
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
