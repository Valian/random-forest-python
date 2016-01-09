import collections

from utils import sample_with_replacement


class RandomForestCreator(object):

    def __init__(self, tree_creator, with_replacement=True, sample_size=0.6):
        self.sample_size = sample_size
        self.with_replacement = with_replacement
        self.tree_creator = tree_creator

    def create_random_forest(self, nr_of_trees, data, attributes, target_attr):
        forest = RandomForest()
        for i in xrange(nr_of_trees):
            sample_data = sample_with_replacement(data, int(len(data) * self.sample_size + 1), self.with_replacement)
            tree = self.tree_creator.create_decision_tree(sample_data, attributes, target_attr)
            forest.add_tree(tree)
        return forest


class RandomForest(object):

    def __init__(self):
        self.trees = []

    def add_tree(self, tree):
        self.trees.append(tree)

    def classify(self, record):
        results = collections.defaultdict(lambda: 0)
        for tree in self.trees:
            single_result = tree.classify(record)
            results[single_result] += 1
        return max(results.keys(), key=lambda key: results[key])

    def classify_many(self, data):
        for record in data:
            yield self.classify(record)

    def __str__(self):
        return '\n'.join(['Tree number {0}: \n{1}\n'.format(i, tree) for i, tree in enumerate(self.trees)])


class DecisionTreeCreator(object):

    def __init__(self, fitness_func, max_depth=None):
        self.max_depth = max_depth
        self.fitness_func = fitness_func

    def create_decision_tree(self, data, attributes, target_attr, actual_depth=0):
        """
        Returns a new decision tree based on the examples given.
        :type data: list[dict[str:str]]
        :type attributes: list[str]
        :type target_attr: str
        :type actual_depth: int
        """
        all_values_for_attribute = [record[target_attr] for record in data]
        most_common_in_children = max(set(all_values_for_attribute), key=all_values_for_attribute.count)
        reached_max_depth = self.max_depth is not None and actual_depth >= self.max_depth
        if not data or (len(attributes) - 1) <= 0 or reached_max_depth:
            return DecisionTreeNode(target_attr, value=most_common_in_children)
        elif len(set(all_values_for_attribute)) == 1:
            return DecisionTreeNode(target_attr, value=all_values_for_attribute[0])
        else:
            best = max(attributes, key=lambda a: self._calculate_fitness(data, a, target_attr))
            tree = DecisionTreeNode(best, most_common_in_children)
            unique_values = set(record[best] for record in data)
            for val in unique_values:
                examples = filter(lambda record: record[best] == val, data)
                attributes = [attr for attr in attributes if attr != best]
                subtree = self.create_decision_tree(examples, attributes, target_attr, actual_depth + 1)
                tree.add_child(subtree, val)
        return tree

    def _calculate_fitness(self, data, attr, target_attr):
        if attr == target_attr:
            return 0.0
        return self.fitness_func(data, attr, target_attr)


class DecisionTreeNode(object):

    def __init__(self, attr_name, default=None, value=None):
        self.default = default
        self.attr_name = attr_name
        self.value = value
        self._children = {}

    def add_child(self, node, attr_value):
        assert node.value not in self._children, "Already in subtree"
        self._children[attr_value] = node

    def is_leaf(self):
        return len(self._children) == 0

    def classify(self, record):
        if self.is_leaf():
            return self.value
        attr_value = record.get(self.attr_name)
        assert attr_value is not None
        child = self._children.get(attr_value)
        if child is None:
            return self.default
        return child.classify(record)

    def classify_many(self, data):
        return map(self.classify, data)

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        children = ['--> {0}:\n\t{1}'.format(val, str(child).replace('\n', '\n\t')) for val, child in self._children.iteritems()]
        return 'Attr: {0}\n{1}'.format(self.attr_name, '\n'.join(children))
