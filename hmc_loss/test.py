#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-21

import unittest
import numpy as np
import networkx as nx
import time
from itertools import product
from hmc_loss import *

class TestHmcLoss(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph()
        h1 = ["A"]
        h2 = ["B", "C", "D", "E"]
        h3 = ["F", "G"]
        h4 = ["H"]
        h5 = ["I", "J", "K"]
        h6 = ["L"]
        self.graph.add_nodes_from(h1+h2+h3+h4+h5)
        self.graph.add_edges_from(list(product(h2, h1))+
                           list(product(h4,["B"]))+
                           list(product(h3,["C"]))+
                           list(product(h5,["E"]))+
                           list(product(h6,["I"]))+
                           list(product(h6,["D"])))
        self.label_list = h1+h2+h3+h4+h5+h6
        self.cost_dict = {'A':1, 'B':1/4, 'C':1/4, 'D':1/4, 'E':1/4, 'F':1/8, 'G':1/8, 'H':1/4, 'I':1/12, 'K':1/12, 'J':1/12, 'L':(1/12)+(1/4)}
        self.cost_list  = [1, 1/4, 1/4, 1/4, 1/4, 1/8, 1/8, 1/4, 1/12, 1/12, 1/12, (1/12)+(1/4)]

    def test_hmc_loss_speed(self):
        print("\nCalculation time test begins")
        data_sizes = [25, 50, 100]
        node_sizes = [25, 50, 100]
        for data_size in data_sizes:
            for node_size in node_sizes:
                true_label = np.random.randint(2, size=(data_size,node_size))
                pred_label = np.random.randint(2, size=(data_size,node_size))
                graph = nx.gnc_graph(node_size)
                label_list = list(range(node_size))
                start_time = time.time()
                cost_list = get_cost_list(graph, 0, label_list)
                cost_list_time = time.time()
                hmc_loss(true_label,
                        pred_label,
                        graph,
                        0,
                        label_list,
                        cost_list
                        )
                end_time = time.time()
                whole_calc_time = end_time - start_time
                cost_calc_time = cost_list_time - start_time
                hmc_calc_time = end_time - cost_list_time
                print("Whole calc. time:{0}, Cost list calc. time:{1}, HMC-loss calc. time:{2}, data size:{3}, node size:{4}".format(whole_calc_time, cost_calc_time, hmc_calc_time, data_size, node_size))
        return 0

    def test_hmc_loss(self):
        y1 = np.array([[1,0,1,0,0,1,0,0,0,0,0,0]])
        y2 = np.array([[1,0,1,0,0,0,1,0,0,0,0,0]])

        #Check symmetric
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, alpha=1, beta=1, average="macro"),
                1/4)
        self.assertEqual(
                hmc_loss(y2, y1, self.graph, "A", self.label_list, self.cost_list, alpha=1, beta=1, average="macro"),
                1/4)

        # Check alpha variable
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, alpha=2, beta=1, average="macro"),
                (1/4) + (1/8))
        # Check beta variable
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, alpha=1, beta=3, average="macro"),
                (1/8) + (3/8))

        # Check dag
        y1 = np.array([[1,0,0,0,1,0,0,0,1,0,0,1]])
        y2 = np.array([[1,0,0,1,0,0,0,0,0,0,0,0]])
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, alpha=1, beta=1, average="macro"),
                (1/4)+(1/4))

        # Check multi input
        y1 = np.array([
            [1,0,0,0,1,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,0,0,0,0],
            [1,0,0,1,0,0,0,0,0,0,0,0]])
        y2 = np.array([
            [1,0,0,0,1,0,0,0,0,1,1,0],
            [1,1,0,0,0,0,0,1,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0]])
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, alpha=1, beta=1, average="macro"),
                (((1/12)+(1/12))+(0)+(1/4))/len(y1))

        # Check gamma drived value(macro)
        gamma = np.array([10/2, 9/3, 10/2])
        beta = 2 / (1+gamma)
        alpha = 2 - beta
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, average="macro"),
                ((beta[0]*(1/12)+beta[0]*(1/12))+(0)+alpha[2]*(1/4))/len(y1))

        # Check gamma drived value(micro)
        gamma = 29 / 7
        beta =  2 / (1+gamma)
        alpha = 2 - beta
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, self.cost_list, average="micro"),
                alpha * np.sum(np.array([1/12])) + beta * np.sum([1/36, 1/36]))

        # Check incomplete matrix
        y1 = np.array([
            [1,0,0,0,1,0],
            [1,0,0,0,0,0],
            [1,0,0,0,1,1]])
        y2 = np.array([
            [1,0,0,0,0,0],
            [1,0,0,0,1,0],
            [1,0,0,0,1,0]])
        label_list = ["A", "B", "C", "D", "E", "F"]
        cost_list  = [1, 1/4, 1/4, 1/4, 1/4, 1/8]
        gamma = (18 - 6) / 6
        beta = 2/ (1+gamma)
        alpha = 2 - beta
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", label_list, cost_list, average="micro"),
                alpha * np.sum(np.array([1/4/3, 1/8/3])) + beta * np.sum([1/4/3])
                )
        return 0

    def test_remove_matrix_redunduncy(self):
        matrix  = np.array([
            [0, 0,  0,  1],
            [0, 0,  1, -1],
            [0, 0, -1,  0],
            [1, 1,  0,  0]])
        removed = np.array([
            [0, 0,  0,  1],
            [0, 0,  1,  0],
            [0, 0, -1,  0],
            [1, 0,  0,  0]])
        graph = nx.DiGraph()
        label_list = ["A", "B", "C", "D"]
        graph.add_nodes_from(label_list)
        graph.add_edges_from(list(product(["B", "C"], ['A']))+
                list(product(["D"], ["B", "C"])))
        self.assertEqual(
                np.array_equal(remove_matrix_redunduncy(matrix, label_list, graph), removed),
                True)

    def test_get_parent_index_list(self):
        self.assertEqual(get_parent_index_list(self.graph, self.label_list),
                [[],[0],[0],[0],[0],[2],[2],[1],[4],[4],[4],[3,8]])
        return 0

    def test_validate_root(self):
        # Check right direction graph
        self.assertEqual(validate_root(self.graph, "A"), 0)
        # Check wrong direction graph
        graph = nx.DiGraph()
        h1 = ["A"]
        h2 = ["B", "C", "D", "E"]
        h3 = ["F", "G"]
        h4 = ["H"]
        h5 = ["I", "J", "K"]
        h6 = ["L"]
        graph.add_nodes_from(h1+h2+h3+h4+h5)
        graph.add_edges_from(list(product(h1, h2)) +
                           list(product(["B"], h4)) +
                           list(product(["C"], h3)) +
                           list(product(["E"], h5)) +
                           list(product(["I"], h6)) +
                           list(product(["D"], h6))) 
        self.assertRaises(ValueError,
                validate_root,
                graph,
                "A")
        return 0

    def test_get_node_cost(self):
        for node in self.label_list:
            self.assertEqual(get_node_cost(self.graph, node, self.cost_dict), self.cost_dict[node])
        return 0

    def test_get_cost_dict(self):
        for node in self.label_list:
            self.assertEqual(get_cost_dict(self.graph, "A")[node], self.cost_dict[node])
        return 0

    def test_get_gamma(self):
        y1 = np.array([
            [1,0,0,0,1,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,0,0,0,0],
            [1,0,0,1,0,0,0,0,0,0,0,0]])
        self.assertEqual(np.array_equal(get_gamma(y1, "macro"), np.array([5,3,5])), True)
        self.assertEqual(get_gamma(y1, "micro"), 29/7)

    def test_get_alpha_beta(self):
        gamma = np.array([5,3,5])
        self.assertEqual(np.array_equal(get_alpha_beta(gamma)[0], np.array([5/3, 3/2, 5/3])), True)
        self.assertEqual(np.array_equal(get_alpha_beta(gamma)[1], np.array([1/3, 1/2, 1/3])), True)
        gamma = 29/7
        self.assertEqual(get_alpha_beta(gamma), (2 - (2/(1+29/7)), 2/(1+29/7)))

if __name__ =='__main__':
    unittest.main()
