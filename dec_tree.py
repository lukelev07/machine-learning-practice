import sys
import math

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

# returns None if shouldn't stop, and the classification of the observations
# otherwise
def should_stop(classifications):
    if len(set(classifications)) is 1: # if all classified as the same
        return classifications[0]
    return None

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
    num_spam = len(filter(lambda x: x, classifications))
    probabilities = [num_spam/num_total, (num_total - num_spam)/num_total]
    return -sum(map(lambda x: x*math.log(x) if x != 0 else 0, probabilities))

def extract_feature_values(observations):
    return map(lambda x: x[0], observations)

def extract_classification_values(observations):
    return map(lambda x: x[1], observations)

def build_decision_tree(observations):
    feature_values = extract_feature_values(observations)
    classification_values = extract_classification_values(observations)
    classification = should_stop(classification_values)
    if classification is not None:
        return DecisionNode(None, None, classification)

    best_feature = None
    best_threshold = None
    best_information_gain = 0
    for feature in range(len(feature_values[0])):
        thresholds = list(set(sorted(map(lambda x: x[feature], feature_values)))) # the value of the specific feature for each observation
        for i in range(len(thresholds)-1):
           threshold = (thresholds[i] + thresholds[i+1]) / 2.0 
           information_gain = calc_information_gain(observations, feature, threshold)
           if information_gain > best_information_gain:
               best_feature = feature
               best_threshold = threshold

    left_observations = filter(lambda x: x[0][best_feature] <= best_threshold, observations)
    right_observations = filter(lambda x: x[0][best_feature] > best_threshold, observations)

    return DecisionNode(build_decision_tree(left_observations), build_decision_tree(right_observations), (feature, threshold))

def print_help():
    print("USAGE: {0} [FEATURES_PATH] [CLASSICATIONS_PATH]".format(sys.argv[0]))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print_help()
        sys.exit()

    features_file = open(sys.argv[1])
    classifications_file = open(sys.argv[2])

    features = map(lambda line: map(float, line.strip().split(',')), features_file.readlines())
    classifications = map(lambda line: bool(float(line.strip().split(',')[0])), classifications_file.readlines())
    observations = zip(features, classifications)
    
    tree = build_decision_tree(observations)
    print(tree)
