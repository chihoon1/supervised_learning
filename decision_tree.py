'''
Implementation of decision tree classifier (supervised learning)
'''



import pandas as pd
import numpy as np
from binary_tree import *


class DecisionTreeClassifier():
    # binary decision tree classifier

    def __init__(self, D, class_labels, is_categorical=False, **kwargs):
        # param: D(pd.DataFrame) is a data set. class_labels(pd.DataFrame) is a class label(y) for each point xj in D
        # param: is_categorical(Bool) true if D based on categorical attributes
        # Merge D and class_labels. The last column of data_set will contain the class labels for each data point
        if D.shape[0] != class_labels.shape[0]:
            raise IndexError("Data and Class Label must have the same number of rows")
        data_set = D.copy()
        #print(class_labels.iloc[:,0])
        data_set.insert(loc=len(data_set.columns), column='classes', value=class_labels.iloc[:,0])
        self.data_set = data_set  # class labels of each data point inserted in the last column
        self.is_categorical = is_categorical
        self.class_domain = list(set(class_labels.iloc[:, 0]))
        self.decision_tree = BinaryTree()
        self.is_trained = False


    def calculate_entropy(self, nvi_lst):
        # Compute shannon entropy of nvi_lst
        # param nvi_lst: list containing num of counts of attribute Xj <=v(split point) and xj=Class i in the data set
        # return an entropy value (float)
        try:
            total_pts_counts = sum(nvi_lst)  # total number of data points in the Data
            entropy = 0
            for i in range(len(nvi_lst)):
                prob_class_i_given_D = nvi_lst[i] / total_pts_counts
                entropy += prob_class_i_given_D * log2(prob_class_i_given_D)
        except ZeroDivisionError:
            # this happens when computing the entropy of data set with no data point
            # In this case, I assumed the entropy would be zero as it shouldn't affect the result of the algorithm
            entropy = 0
        return entropy * -1


    def get_info_gain(self, nvi_lst, nci_lst):
        # Compute information gain of dataset with the given split
        # param nvi_lst: list containing num of counts of attribute Xj <=v(split point) and xj=Class i in the data set
        # param nci_lst: list containing num of counts of xj=Class i in the data set(can be splitted data set)
        # return an information gain (int)
        D_pts_counts = sum(nci_lst)
        Y_pts_counts = sum(nvi_lst)  # num of pts <= split point (== split point if categorical var)
        N_pts_counts = D_pts_counts - Y_pts_counts  # num of pts >= split point (== split point if categorical var)
        entroy_D = self.calculate_entropy(nci_lst)
        entroy_D_Y = self.calculate_entropy(nvi_lst)
        nvi_complement_lst = [nci_lst[i] - nvi_lst[i] for i in range(len(nci_lst))]
        entroy_D_N = self.calculate_entropy(nvi_complement_lst)
        info_gain = entroy_D - (Y_pts_counts/D_pts_counts*entroy_D_Y) - (N_pts_counts/D_pts_counts*entroy_D_N)
        return info_gain


    def search_optimal_numeric_split_point(self, attr_index):
        # find the best split point for real value variable
        # param attr_index(int): indicates the column index of attribute where we want to find the best split point
        # return: best split point(int) and information gain(float) of the split point
        modified_D = self.data_set.loc[:, [self.data_set.columns[attr_index], 'classes']]
        Xj_attribute_class_lst = [[modified_D.iloc[i,0],modified_D.iloc[i,1]] for i in range(len(modified_D))]
        Xj_attribute_class_lst.sort(key=lambda x: x[0])  # 2d list. first column: attribute, second column: class
        midp_lst = []
        all_n_v_class_i_lst = []  # list of lists containing nvi for all midpoints v
        # list containing the counts of data points in class i for all classes in data set
        n_class_i_lst = [0] * len(self.class_domain)
        # count the number of data points associate with the class i and <= a particular value in the numeric attribute
        for k in range(len(Xj_attribute_class_lst)):  # loop through all data points except for the last data point
            if k < len(Xj_attribute_class_lst) - 1:
                midpoint = (Xj_attribute_class_lst[k][0] + Xj_attribute_class_lst[k+1][0])/2
            n_class_i_lst[self.class_domain.index(Xj_attribute_class_lst[k][1])] += 1
            if k > 0 and Xj_attribute_class_lst[k][0] <= midp_lst[-1]:
                # this happens when two same values of attribute Xj exist
                all_n_v_class_i_lst[-1] = n_class_i_lst[:]
            if k == 0 or midp_lst[-1] != midpoint:  # midpoint must be unique
                midp_lst.append(midpoint)
                all_n_v_class_i_lst.append(n_class_i_lst[:])
        # find the best split point
        optimal_split_point, best_score = None, -float('inf')
        for k in range(len(midp_lst)):
            score = self.get_info_gain(all_n_v_class_i_lst[k], n_class_i_lst)
            # print(f"{attr_index}-th column split point {midp_lst[k]} information gain: {score}")  # debugging purpose
            if best_score < score:
                # new best split point found
                optimal_split_point = midp_lst[k]
                best_score = score
        return optimal_split_point, best_score


    def search_optimal_categorical_split_point(self, attr_index):
        # find the best split point for categorical variable
        # param attr_index(int): indicates the column index of attribute where we want to find the best split point
        # return: best split point(int) and information gain(float) of the split point
        modified_D = self.data_set.loc[:, [self.data_set.columns[attr_index], 'classes']]
        Xj_attribute_class_lst = [[modified_D.iloc[i, 0], modified_D.iloc[i, 1]] for i in range(len(modified_D))]
        # list containing the counts of data points in class i for all classes in data set
        n_class_i_lst = [0] * len(self.class_domain)
        all_n_v_class_i_dict = {}  # dictionary containing as value nvi for all categories vi in dom(Xj), (key: vi)
        # count the number of data points associate with the class i and a particular value in the categorical attribute
        for i in range(len(Xj_attribute_class_lst)):
            n_class_i_lst[self.class_domain.index(Xj_attribute_class_lst[i][1])] += 1
            if all_n_v_class_i_dict.get(Xj_attribute_class_lst[i][0]) is None:
                all_n_v_class_i_dict[Xj_attribute_class_lst[i][0]] = [0]*len(self.class_domain)
        for i in range(len(Xj_attribute_class_lst)):
            class_index = self.class_domain.index(Xj_attribute_class_lst[i][1])
            category_attr_val = Xj_attribute_class_lst[i][0]
            all_n_v_class_i_dict[category_attr_val][class_index] += 1
        # find the best split point
        optimal_split_point, best_score = None, -float('inf')
        for key, val in all_n_v_class_i_dict.items():
            score = self.get_info_gain(val, n_class_i_lst)
            # print(f"{attr_index}-th column split point {key} information gain: {score}")  # debugging purpose
            if best_score < score:
                # new best split point found
                optimal_split_point = key
                best_score = score
        return optimal_split_point, best_score


    def paritioning_regions(self, D, purity_threshold, num_pts_threshold, parent_split_pt=None, **kwargs):
        # param: D is data set(or Region/splitted data set). Expect to be pandas DataFrame
        # param: purity_threshold indicates that if max purity of D <= this threshold, then leaf node created
        # param: num_pts_threshold(int, default=0) indicates that if |D| is >= this threshold, then leaf node created
        # param: parent_split_pt(tuple) is a split point and its attribute column index of parent region
        # return list containing split points in the order of Depth-First
        # print(f"Parent split point: {parent_split_pt}")  # debugging purpose
        n = len(D)
        # list containing the counts of data points in class i for all classes in data set
        n_class_i_lst = [0] * len(self.class_domain)
        for i in range(len(D)):
            n_class_i_lst[self.class_domain.index(D.iloc[i, -1])] += 1

        max_purity, c_star = -1, None  # c_star is the class with the max purity
        # find the class that has the highest purity
        for i in range(len(n_class_i_lst)):
            if max_purity < n_class_i_lst[i]:
                max_purity = n_class_i_lst[i]
                c_star = self.class_domain[i]
        if n <= num_pts_threshold or max_purity / n >= purity_threshold:  # base case
            # stop going further recursion and create a leaf node here
            return BTreeNode((c_star, -1))  # -1 denotes this node is a leaf node
        # find the best split attribute and point
        best_split_pt, max_score, its_attr_type = None, -float('inf'), None
        for col_j in range(D.shape[1] - 1):
            attr_type = str(D.iloc[:,col_j].dtype)
            if attr_type[:3] in ('int', 'flo'):  # numeric variable
                split_pt, score = self.search_optimal_numeric_split_point(col_j)
            else:
                split_pt, score = self.search_optimal_categorical_split_point(col_j)
            if max_score < score and ((split_pt, col_j) != parent_split_pt):
                best_split_pt = (split_pt, col_j)
                max_score = score
                its_attr_type = attr_type
        quot = "'" if type(best_split_pt[0]) == str else ""
        yes_expr = "==" if self.is_categorical else "<="
        no_expr = "!=" if self.is_categorical else ">"
        query1 = f"{D.columns[best_split_pt[1]]} {yes_expr} {quot}{best_split_pt[0]}{quot}"
        query2 = f"{D.columns[best_split_pt[1]]} {no_expr} {quot}{best_split_pt[0]}{quot}"
        D_Y = D.query(query1)
        D_N = D.query(query2)
        node = BTreeNode(best_split_pt, dtype=its_attr_type)
        if self.decision_tree.root is None:
            self.decision_tree.root = node
        node.left_child = self.paritioning_regions(D_Y, purity_threshold, num_pts_threshold, best_split_pt)
        node.right_child = self.paritioning_regions(D_N, purity_threshold, num_pts_threshold, best_split_pt)
        return node


    def train(self, purity_threshold, num_pts_threshold=0):
        # param: purity_threshold indicates that if max purity of D <= this threshold, then leaf node created
        # param: num_pts_threshold(int, default=0) indicates that if |D| is >= this threshold, then leaf node created
        # return list containing split points in the order of Depth-First
        self.paritioning_regions(self.data_set, purity_threshold, num_pts_threshold)
        self.is_trained = True
        return self.decision_tree


    def predict(self, x):
        # predict x's class with the trained decision tree classifier
        # param x: a data point (1D array/vector-like)
        # return predicted class of x (str)
        curr_node = self.decision_tree.root
        while curr_node is not None:
            if curr_node.node[1] == -1:  # node is a leaf node
                return curr_node.node[0]  # return the predicted class
            attr_type = curr_node.data_info['dtype']
            if attr_type[:3] in ('int', 'flo'):  # numeric variable
                print(f"child: {curr_node.left_child}, {curr_node.right_child}")
                if x[curr_node.node[1]] <= curr_node.node[0]:  # yes to split point (left)
                    curr_node = curr_node.left_child
                else:  # no to split point (right)
                    curr_node = curr_node.right_child
            else:  # categorical variable
                if x[curr_node.node[1]] == curr_node.node[0]:  # yes to split point (left)
                    curr_node = curr_node.left_child
                else:  # no to split point (right)
                    curr_node = curr_node.right_child
        else:  # node is None which can only happen at root level and this implies the model is not trained
            raise Exception("Tree is not trained")



if __name__ == '__main__':
    price = [10., 10., 20., 10., 10., 10., 20., 20.]
    chef = ['A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    class_quality = ['L', 'L', 'H', 'H', 'H', 'H', 'L', 'H']
    D = pd.DataFrame({'Price': price, 'Chef': chef})
    class_labels = pd.DataFrame({'Quality': class_quality})

    DTC = DecisionTreeClassifier(D, class_labels, is_categorical=True)
    DTC.train(purity_threshold=0.75)
    if DTC.is_trained:
        DTC.decision_tree.BFS_traverse(edge_labels=('Yes', 'No'))
    else:
        print("Not Trained")

    # predict a class label of a random point
    x = []
    rand_arr = np.random.normal(0, 1, size=2)
    x.append(int(25 * np.abs(rand_arr[0])))
    x.append('A' if rand_arr[1] >= 0 else 'B')
    print(f"x: {x}")
    print(f"Predicted class of x: {DTC.predict(x)}")
    x = [9, 'A']
    print(f"x: {x}")
    print(f"Predicted class of x: {DTC.predict(x)}")