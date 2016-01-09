"""
This module holds functions that are responsible for creating a new
decision tree and for using the tree for data classificiation.
"""
import collections

def get_classification(record, tree):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if isinstance(tree, str):
        return tree
    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)


def classify(tree, data):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []
    
    for record in data:
        classification.append(get_classification(record, tree))

    return classification


def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        # We use the collections.defaultdict function to add a function to the
        # new tree that will be called whenever we query the tree with an
        # attribute that does not exist.  This way we return the default value
        # for the target attribute whenever, we have an attribute combination
        # that wasn't seen during training.
        tree = {best:collections.defaultdict(lambda: default)}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree
