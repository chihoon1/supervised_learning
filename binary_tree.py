'''
Implementation of Binary Tree
used for decision tree classfication
'''



from math import log2, ceil
from queue import Queue


class BTreeNode():  # class for a node of binary tree
    def __init__(self, instance, **kwargs):
        self.node = instance
        self.left_child = None
        self.right_child = None
        self.data_info = kwargs  # any other custom information about data will be stored here

    def __repr__(self):
        return (str(self.node))


class BinaryTree():  # class representing a binary tree
    def __init__(self, root=None):
        root_node = BTreeNode(root) if root is not None else None
        self.root = root_node
        # self.size = 1


    def BFS_traverse(self, **kwargs):
        # print out Binary Tree in BFS traversal order
        edge_labels = kwargs.get("edge_labels", ('Left', 'Right'))  # name for an edge. Ex) left edge or right edge
        unvisited = Queue()
        unvisited.put(self.root)
        nodes_in_the_same_level = []

        while not unvisited.empty():
            curr_node = unvisited.get()
            nodes_in_the_same_level.append([curr_node])
            if curr_node is not None:
                unvisited.put(curr_node.left_child)
                unvisited.put(curr_node.right_child)
        count = 0
        exp_of_two = 1
        _ = []
        for i in range(len(nodes_in_the_same_level)):  # print the nodes of the tree in BFS order
            count += 1
            _.append(nodes_in_the_same_level[i])
            if count == exp_of_two:
                print(f"level {log2(exp_of_two)}")
                for j in range(len(_)):
                    #yes_or_no = "Yes" if j % 2 == 0 else "No"
                    yes_or_no = edge_labels[j % 2]
                    if exp_of_two == 1: yes_or_no = 'Root'  # root node case
                    print(f"{yes_or_no}: {_[j]}")
                count = 0
                exp_of_two *= 2
                _ = []
        print(f"level {ceil(log2(exp_of_two))}")  # leaf level
        for j in range(len(_)):
            #yes_or_no = "Yes" if j % 2 == 0 else "No"
            yes_or_no = edge_labels[j % 2]
            print(f"{yes_or_no}: {_[j]}")