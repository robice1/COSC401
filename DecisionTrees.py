import collections
import math
class DTNode:
    def __init__(self, decision):
        self.data = None
        self.decision = decision
        self.children = []

    def predict(self, input_object):
        if callable(self.decision):
            child_index = self.decision(input_object)
            child_node = self.children[child_index]
            if child_node is None:
                raise ValueError("Invalid decision")
            return child_node.predict(input_object)
        else:
            return self.decision
    
    def leaves(self, next_node=None):
        if not next_node:
            next_node = self
        if len(next_node.children) == 0:
            return 1
        num_leaves = 0
        for next_leaf in next_node.children:
            num_leaves += self.leaves(next_leaf)
        return num_leaves

def partition_by_feature_value(dataset, feature_index):
    # Get unique feature values
    unique_values = list(set([data[0][feature_index] for data in dataset]))
    
    # Create a mapping of feature values to indices
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    
    # Initialize partitioned dataset
    partitioned_dataset = [[] for _ in range(len(unique_values))]

    def separator(value):
        return value_to_index[value[feature_index]]
    
    # Partition the dataset based on feature values
    for value, label in dataset:
        index = separator(value)
        partitioned_dataset[index].append((value, label))
    
    return separator, partitioned_dataset

def misclassification(dataset):
    impurity = 1 - max(proportion(dataset))
    return impurity

def gini(dataset):
    impurity = sum([(1 - p_k) * p_k for p_k in proportion(dataset)])
    return impurity

def entropy(dataset):
    impurity = -sum([p_k * math.log(p_k, 2) for p_k in proportion(dataset)])
    return impurity

def proportion(dataset):
    labels = list(set([item[1] for item in dataset]))
    m = [0 for _ in range(len(labels))]
    label_dict = {}
    for label in range(len(labels)):
        label_dict.update({labels[label]:label})
    for i in dataset:
        m[label_dict[i[1]]] += 1
    return [i / len(dataset) for i in m]


def get_labels(dataset):
    """
    :param dataset: the datas
    :return: set of labels from dataset
    """
    set_of_labels = []
    for label_index in range(len(dataset)):
        set_of_labels.append(dataset[label_index][1])
    return set(set_of_labels)

def choose_feature(dataset, features):
    """
    :param dataset: the datas
    :param features: list of features
    :return: result, feature, data
    """
    feature_result = None
    data_result = None
    minimum = float('inf')
    for label in features:
        smallest = 0
        separator, partitioned_data = partition_by_feature_value(dataset, label)
        impurities = []
        for impurity in partitioned_data:
            impurities.append(misclassification(impurity))
        for index in range(len(impurities)):
            smallest += (len(partitioned_data[index]) / len(dataset)) * impurities[index]
        if smallest < minimum:
            minimum = smallest
            feature_result = label
            data_result = partitioned_data
    return feature_result, separator, data_result

def most_common_label(data):
    """
    return most common label in dataset
    :param data: dataset
    :return: most common label
    """
    label = collections.Counter([x[1] for x in data]).most_common(1)[0][0]
    return label

def make_decision(index, data):
    """
    :param index: index to check
    :param data: data set
    :return: list of decisions
    """
    listy = []
    for item in data:
        listy.append(item[0][index])
    decisions = list(set(listy))
    return decisions


def train_tree(dataset, criterion, attribute_list=None):
    """
    :param dataset: list of pairs, where the first element in each pair is a feature vector,
                    and the second is a classification.
    :param criterion: function evaluates a dataset for a specific impurity measure
    :param attribute_list: function recursive passing attribute list, empty for first call
    :return: node
    """
    labels = get_labels(dataset)
    common = most_common_label(dataset)
    if attribute_list is None:
        attribute_list = [i for i in range(len(dataset[0][0]))]
    if len(labels) == 1 or not len(attribute_list):
        return DTNode(common)
    else:
        feature_result, _, data_result = choose_feature(dataset, attribute_list)
        outcomes = make_decision(feature_result, dataset)
        node = DTNode(lambda x: outcomes.index(x[feature_result]))
        attribute = [attribute for attribute in attribute_list if attribute != feature_result]
        node.children = [train_tree(data, criterion, attribute) for data in data_result]
        return node


def main():
    dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False)
    ]
    t = train_tree(dataset, misclassification)
    print(t.predict((True, False)))
    print(t.predict((False, False)))


if __name__ == '__main__':
    main()