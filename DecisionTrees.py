class DTNode:
    def __init__(self, decision):
        self.decision = decision
        self.children = {}

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

def main():
    # The following (leaf) node will always predict True
    node = DTNode(True)

    # Prediction for the input (True, False):
    print(node.predict((True, False)))

    # Sine it's a leaf node, the input can be anything. It's simply ignored.
    print(node.predict(None))

    yes_node = DTNode("Yes")
    no_node = DTNode("No")
    tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
    tree_root.children = [yes_node, no_node]

    print(tree_root.predict((False, 'Red', 3.5)))
    print(tree_root.predict((False, 'Green', 6.1)))

    n = DTNode(True)
    print(n.leaves())
    t = DTNode(True)
    f = DTNode(False)
    n = DTNode(lambda v: 0 if not v else 1)
    n.children = [t, f]
    print(n.leaves())


if __name__ == '__main__':
    main()