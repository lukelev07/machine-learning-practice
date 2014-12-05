import sys
import math
import random
from collections import defaultdict, Counter

class DecisionNode(object):
    # if the Node is a leaf, the label represents the classification
    # if the Node isn't a leaf, the label represents the decision metric,
    # which is represented as a tuple (feature, threshold)
    def __init__(self, left, right, label):
        self.left = left
        self.right = right
        self.label = label

    # TODO: this doesn't work
    def __str__(self):
        return '''    {1}
                    /    \
                  /       \
                {0}         {2}
               '''.format(self.left, self.label, self.right)

def max_child_depth(node):
    if node is None:
        return 0
    return 1 + max(max_child_depth(node.left), max_child_depth(node.right))

# returns None if shouldn't stop, and the classification of the observations
# otherwise
def should_stop(classifications, depth):
    if len(set(classifications)) is 1: # if all classified as the same
        return classifications[0]
    if depth > 30:
        return majority(classifications)
    if entropy(classifications) < .005:
        return majority(classifications)
    return None

def majority(elems):
    c = Counter(elems)
    return c.most_common()[0]

def calc_information_gain(observations, feature, threshold):
    left_observations = filter(lambda x: x[0][feature] <= threshold, observations)
    right_observations = filter(lambda x: x[0][feature] > threshold, observations)

    together = extract_classification_values(observations)
    left = extract_classification_values(left_observations)
    right = extract_classification_values(right_observations)

    fraction_left = len(left) / float(len(together))
    fraction_right = len(right) / float(len(together))

    return entropy(together) - (fraction_left * entropy(left) + fraction_right * entropy(right))

# we assume that there's only two classifications, although this could be
# extended to be more robust
def entropy(classifications):
    num_total = float(len(classifications))
    if int(num_total) is 0:
        return 0
    num_spam = len(filter(lambda x: x is '1', classifications))
    probabilities = [num_spam/num_total, (num_total - num_spam)/num_total]
    return -sum(map(lambda x: x*math.log(x, 2) if x != 0 else 0, probabilities))

def extract_feature_values(observations):
    return map(lambda x: x[0], observations)

def extract_classification_values(observations):
    return map(lambda x: x[1], observations)

def build_decision_tree(observations, depth):
    feature_values = extract_feature_values(observations)
    classification_values = extract_classification_values(observations)
    classification = should_stop(classification_values, depth)
    if classification is not None:
        return DecisionNode(None, None, classification)

    all_features = range(len(feature_values[0]))
    best_feature = None
    best_threshold = None
    best_information_gain = -1

    while best_information_gain is -1:
        features = random.sample(all_features, int(math.sqrt(len(all_features))))
        for feature in features:
            thresholds = list(set(sorted(map(lambda x: x[feature], feature_values)))) # the value of the specific feature for each observation
            for i in range(len(thresholds)-1):
               threshold = (thresholds[i] + thresholds[i+1]) / 2.0 
               information_gain = calc_information_gain(observations, feature, threshold)
               if information_gain > best_information_gain:
                   best_feature = feature
                   best_threshold = threshold
                   best_information_gain = information_gain

    left_observations = filter(lambda x: x[0][best_feature] <= best_threshold, observations)
    right_observations = filter(lambda x: x[0][best_feature] > best_threshold, observations)

    return DecisionNode(build_decision_tree(left_observations, depth+1),
            build_decision_tree(right_observations, depth+1), (best_feature, best_threshold))

def classify_from_tree(feature, decision_tree):
    if decision_tree.left is None and decision_tree.right is None:
        return decision_tree.label
    decider = decision_tree.label
    if feature[decider[0]] <= decider[1]:
        return classify_from_tree(feature, decision_tree.left)
    return classify_from_tree(feature, decision_tree.right)

def classify_from_trees(feature, decision_trees):
    votes = defaultdict(lambda: 0)
    for tree in decision_trees:
        vote = classify_from_tree(feature, tree)
        votes[vote] += 1
    return max(votes.keys(), key=lambda vote: votes[vote])

def print_help():
    print("USAGE: {0} [FEATURES_PATH] [CLASSICATIONS_PATH]".format(sys.argv[0]))

def print_differ(val_features_path, val_classifications_path, trees):
    val_features_file = open(val_features_path)
    val_classifications_file = open(val_classifications_path)

    val_features = map(lambda line: map(float, line.strip().split(',')), val_features_file.readlines())
    val_classifications = map(lambda line: line.strip().split(',')[0], val_classifications_file.readlines())
    output = map(lambda x: classify_from_trees(x, trees), val_features)
    differ = 0
    for i in range(len(output)):
        if output[i] is not val_classifications[i]:
            differ += 1
    print("{0} percent was the same".format((len(output)-differ)/float(len(output))*100))

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print_help()
        sys.exit()

    #T = int(sys.argv[1])
    tests = [1,2,5,10,25]
    T = 25
    features_file = open(sys.argv[2])
    classifications_file = open(sys.argv[3])

    features = map(lambda line: map(float, line.strip().split(',')), features_file.readlines())
    classifications = map(lambda line: line.strip().split(',')[0], classifications_file.readlines())
    observations = zip(features, classifications)
    
    trees = []
    for i in xrange(T):
        observations_subset = [random.choice(observations) for _ in range(len(observations)) ]
        # observations_subset = observations
        tree = build_decision_tree(observations_subset, 0)
        print("Built tree number {0}".format(i+1))
        # print("Tree depth: " + str(max_child_depth(tree)))
        trees.append(tree)

        if i+1 in tests:
            print("Classifying for {0} trees".format(i+1))
            print_differ(sys.argv[4], sys.argv[5], trees)


