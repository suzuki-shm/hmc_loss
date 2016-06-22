#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-21
import numpy as np
from queue import Queue

def hmc_loss(true_matrix, pred_matrix, graph, root, label_list, alpha=1, beta=1):
    validate_root(graph, root)
    cost_graph = get_cost_graph(graph, root)
    loss = get_loss(true_matrix, pred_matrix, cost_graph, label_list, alpha, beta)
    return loss

def get_loss(true_matrix, pred_matrix, cost_graph, label_list, alpha, beta):
    cost_list = [cost_graph[node]["cost"] for node in label_list]
    cost_list = np.array(cost_list)

    c_matrix = remove_matrix_redunduncy(true_matrix-pred_matrix, label_list, cost_graph)

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
            if parent != "cost":
                tmp.append(label_list.index(parent))
        tmp = sorted(tmp)
        parent_index.append(tmp)
    return parent_index

def validate_root(graph, root):
    if len(graph.successors(root)) !=0:
        raise ValueError("Graph direction is wrong.",
            "This function requires bottom-up direction.")
    return 0

def get_node_cost(graph, node):
    """
        get cost of input node along
        To adapt for DAG, if parent node doesn't have cost attribution, return 0.
        This node calculate again in the loop.
    """
    cost = 0
    ancestors = graph.successors(node)
    if len(ancestors) == 0:
        return 1
    for ancestor in ancestors:
        if ancestor != "cost":
            try:
                cost += graph[ancestor]["cost"]/len(graph.predecessors(ancestor))
            except KeyError:
                return 0
    return cost

def get_cost_graph(graph, root):
    """
        add cost to input graph by breath first search
    """
    cost_graph = graph.copy()
    q = Queue()
    cost_graph[root]["cost"] = get_node_cost(cost_graph, root)
    for predecessor in cost_graph.predecessors(root):
        q.put(predecessor)
    while q.empty() == False:
        child = q.get()
        cost_graph[child]["cost"] = get_node_cost(cost_graph, child)

        child_predecessors = cost_graph.predecessors(child)
        for predecessor in child_predecessors:
            q.put(predecessor)
    return cost_graph

def main():
    pass

if __name__ =='__main__':
    main()
