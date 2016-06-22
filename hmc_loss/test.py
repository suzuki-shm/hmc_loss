#!/usr/bin/env python
# vim:fileencoding=utf-8
#Author: Shinya Suzuki
#Created: 2016-06-21

import unittest
import numpy as np
import networkx as nx
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

    def test_hmc_loss(self):
        y1 = np.array([[1,0,1,0,0,1,0,0,0,0,0,0]])
        y2 = np.array([[1,0,1,0,0,0,1,0,0,0,0,0]])

        #Check symmetric
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list),
                1/4)
        self.assertEqual(
                hmc_loss(y2, y1, self.graph, "A", self.label_list),
                1/4)

        # Check alpha variable
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, alpha=2),
                (1/4) + (1/8))

        # Check beta variable
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list, beta=3),
                (1/8) + (3/8))

        # Check dag
        y1 = np.array([[1,0,0,0,1,0,0,0,1,0,0,1]])
        y2 = np.array([[1,0,0,1,0,0,0,0,0,0,0,0]])
        self.assertEqual(
                hmc_loss(y1, y2, self.graph, "A", self.label_list),
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
                hmc_loss(y1, y2, self.graph, "A", self.label_list),
                (((1/12)+(1/12))+(0)+(1/4))/len(y1))
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
        self.assertEqual(validate_root(self.graph, "A"), 0)
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
        self.assertEqual(get_node_cost(self.graph, "A"), 1)
        return 0

    def test_get_cost_graph(self):
        self.assertEqual(get_cost_graph(self.graph, "A")["A"]["cost"], 1)
        self.assertEqual(get_cost_graph(self.graph, "A")["B"]["cost"], 1/4)
        self.assertEqual(get_cost_graph(self.graph, "A")["L"]["cost"], (1/12)+(1/4))
        return 0

if __name__ =='__main__':
    unittest.main()
