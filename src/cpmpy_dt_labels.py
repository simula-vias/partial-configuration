# %%
import json

import numpy as np
import pandas as pd
from cpmpy import *
from sklearn.model_selection import train_test_split

from common import load_data


class DecisionTreeTrainerCP:
    def __init__(
        self,
        max_depth=3,
        max_features=None,
        max_classes=None,
        categorical_features=None,
        no_reuse_features=True,
        feature_names=None,
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_classes = max_classes
        self.categorical_features = (
            categorical_features or []
        )  # List of indices of categorical features
        self.no_reuse_features = (
            no_reuse_features  # Whether to prevent feature reuse in paths
        )
        self.model = Model()
        self.feature_names = feature_names
    def create_tree_variables(self, n_features, n_classes, X):
        """Create CPMpy variables for the decision tree structure"""
        self.feature_vars = {}
        self.threshold_vars = {}
        self.categorical_set_vars = {}  # New: variables for categorical feature sets
        self.leaf_prediction_vars = {}
        self.n_classes = n_classes
        self.n_features = n_features

        # Find min and max values for each feature and unique values for categorical features
        feature_ranges = []
        self.categorical_values = {}
        # Track features with no variation
        constant_features = []
        
        for j in range(X.shape[1]):
            if j in self.categorical_features:
                unique_values = np.unique(X[:, j])
                self.categorical_values[j] = unique_values
                # For categorical features, we still need a range for the threshold variables
                feature_ranges.append((0, len(unique_values) - 1))
                # If there's only one unique value, mark as constant
                if len(unique_values) == 1:
                    constant_features.append(j)
            else:
                min_val = X[:, j].min()
                max_val = X[:, j].max()
                feature_ranges.append((min_val, max_val))
                # If min equals max, this feature has no variation
                if min_val == max_val:
                    constant_features.append(j)

        # Create variables for each node in the tree
        for depth in range(self.max_depth):
            for node in range(2**depth):
                node_id = (depth, node)

                # Feature selection variables
                self.feature_vars[node_id] = IntVar(
                    0, n_features - 1, name=f"feature_{depth}_{node}"
                )
                
                # Add constraint to prevent selecting constant features
                for feat_idx in constant_features:
                    self.model += self.feature_vars[node_id] != feat_idx

                # Threshold variable - allow any value between min and max
                min_val = min(r[0] for r in feature_ranges) * 100
                max_val = max(r[1] for r in feature_ranges) * 100
                self.threshold_vars[node_id] = IntVar(
                    int(min_val), int(max_val), name=f"threshold_{depth}_{node}"
                )

                # For categorical features, create set variables
                if self.categorical_features:
                    self.categorical_set_vars[node_id] = {}
                    for feat_idx in self.categorical_features:
                        if feat_idx in constant_features:
                            continue  # Skip constant categorical features
                        n_values = len(self.categorical_values[feat_idx])
                        # Create a boolean variable for each possible value
                        self.categorical_set_vars[node_id][feat_idx] = [
                            BoolVar(name=f"cat_set_{depth}_{node}_{feat_idx}_{val_idx}")
                            for val_idx in range(n_values)
                        ]
                        # Ensure at least one value is selected
                        self.model += any(self.categorical_set_vars[node_id][feat_idx])

                # For non-leaf nodes, ensure threshold is between min and max of selected feature
                if depth < self.max_depth - 1:
                    for feature_idx in range(n_features):
                        if feature_idx in constant_features:
                            continue  # Skip constant features
                        
                        feature_selected = self.feature_vars[node_id] == feature_idx
                        if feature_idx in self.categorical_features:
                            # For categorical features, we don't need to constrain the threshold
                            # as we'll use the set variables instead
                            pass
                        else:
                            feature_min = int(feature_ranges[feature_idx][0] * 100)
                            feature_max = int(feature_ranges[feature_idx][1] * 100)
                            self.model += ~feature_selected | (
                                self.threshold_vars[node_id] >= feature_min
                            )
                            self.model += ~feature_selected | (
                                self.threshold_vars[node_id] <= feature_max
                            )

                # For leaf nodes, create prediction variables for each class
                if depth == self.max_depth - 1:
                    self.leaf_prediction_vars[node_id] = IntVar(
                        0, n_classes - 1, name=f"leaf_{depth}_{node}"
                    )

        # Add constraints to ensure different thresholds for the same feature at different nodes
        # But only for sibling nodes to reduce constraint complexity
        for depth in range(self.max_depth - 1):  # Only for non-leaf nodes
            for node in range(0, 2**depth, 2):  # Process pairs of siblings
                if node + 1 < 2**depth:  # Make sure the sibling exists
                    node_id1 = (depth, node)
                    node_id2 = (depth, node + 1)
                    # If the same feature is selected at both nodes, ensure different thresholds
                    same_feature = self.feature_vars[node_id1] == self.feature_vars[node_id2]
                    diff_threshold = self.threshold_vars[node_id1] != self.threshold_vars[node_id2]
                    self.model += ~same_feature | diff_threshold

        # Add constraint to prevent feature reuse in paths from root to leaf
        if self.no_reuse_features:
            self.add_no_feature_reuse_constraints()

        # Feature usage tracking
        if self.max_features is not None:
            self.feature_used = []
            for i in range(n_features):
                self.feature_used.append(BoolVar(name=f"feature_used_{i}"))

                feature_usage = []
                for node_id in self.feature_vars:
                    feature_usage.append(self.feature_vars[node_id] == i)
                self.model += self.feature_used[i] == any(feature_usage)

            self.model += sum(self.feature_used) <= self.max_features

        # Class usage tracking
        if self.max_classes is not None:
            self.class_used = []
            for i in range(n_classes):
                self.class_used.append(BoolVar(name=f"class_used_{i}"))

                class_usage = []
                for node_id in self.leaf_prediction_vars:
                    class_usage.append(self.leaf_prediction_vars[node_id] == i)
                self.model += self.class_used[i] == any(class_usage)

            self.model += sum(self.class_used) <= self.max_classes

    def add_no_feature_reuse_constraints(self):
        """Add constraints to prevent feature reuse in any path from root to leaf"""
        # For each leaf node, trace back to the root and ensure no feature is used twice
        for leaf_depth in range(self.max_depth - 1, self.max_depth):
            for leaf_node in range(2**leaf_depth):
                leaf_id = (leaf_depth, leaf_node)

                # Trace the path from leaf to root
                path_nodes = []
                current_node = leaf_node
                for d in range(leaf_depth, -1, -1):
                    path_nodes.append((d, current_node))
                    current_node = current_node // 2  # Parent node

                # Add constraints to ensure no feature is used twice in this path
                for i in range(len(path_nodes)):
                    for j in range(i + 1, len(path_nodes)):
                        node_i = path_nodes[i]
                        node_j = path_nodes[j]
                        # Features at nodes i and j must be different
                        self.model += (
                            self.feature_vars[node_i] != self.feature_vars[node_j]
                        )

    def encode_prediction(self, sample, node_id=(0, 0)):
        """Recursively encode the prediction logic for a sample"""
        depth, node = node_id

        if depth == self.max_depth - 1:
            return self.leaf_prediction_vars[node_id]

        # Create a variable for the split condition
        split_condition = BoolVar(name=f"split_{depth}_{node}")

        # Get the feature index for this node
        feature_var = self.feature_vars[node_id]

        # Create a list to store conditions for each feature
        feature_conditions = []

        # For each possible feature
        for feat_idx in range(self.n_features):
            feature_selected = feature_var == feat_idx

            if feat_idx in self.categorical_features:
                # For categorical features, check if the sample's value is in the selected set
                sample_value = sample[feat_idx]
                # Find the index of this value in the unique values list
                value_idx = np.where(self.categorical_values[feat_idx] == sample_value)[0][0]
                # The condition is true if this value's corresponding boolean variable is true
                in_set = self.categorical_set_vars[node_id][feat_idx][value_idx]
                feature_conditions.append(feature_selected & in_set)
            else:
                # For numerical features, use the threshold comparison
                sample_value = int(sample[feat_idx] * 100)
                threshold = self.threshold_vars[node_id]
                feature_conditions.append(feature_selected & (sample_value >= threshold))

        # The split condition is true if the condition for the selected feature is true
        self.model += split_condition == any(feature_conditions)

        left_child = (depth + 1, node * 2)
        right_child = (depth + 1, node * 2 + 1)

        # Create variables for the left and right predictions
        left_pred = self.encode_prediction(sample, left_child)
        right_pred = self.encode_prediction(sample, right_child)

        # Create a new variable for the result
        result = IntVar(0, self.n_classes - 1)

        # Add constraints to enforce the decision logic
        self.model += (result == left_pred) | (result == right_pred)  # Result must be one of the two options

        # Use direct logical constraints instead of implications
        self.model += (~split_condition) | (result == left_pred)  # If condition is true, use left
        self.model += (split_condition) | (result == right_pred)  # If condition is false, use right

        return result

    def add_accuracy_objective(self, X, y):
        """Add accuracy objective for the training data using class labels"""
        # Create variables to track correct predictions for each sample
        correct_predictions = []

        for i, sample in enumerate(X):
            prediction = self.encode_prediction(sample)
            
            # Create a variable that is 1 if prediction is correct, 0 otherwise
            is_correct = BoolVar(name=f"correct_{i}")
            
            # Prediction is correct if it equals the true label
            self.model += is_correct == (prediction == y[i])
            
            correct_predictions.append(is_correct)

        # Total accuracy is the sum of all correct predictions
        total_accuracy = sum(correct_predictions)

        return total_accuracy

    def add_regularization_terms(self):
        """Create regularization terms for the objective function"""
        regularization = 0

        # Add feature selection terms to objective
        for node_id in self.feature_vars:
            depth, _ = node_id
            weight = 2 ** (self.max_depth - depth - 1)
            regularization += weight * self.feature_vars[node_id]
        
        # Add a term to encourage diversity in feature selection, but less aggressively
        # Count how many times each feature is used
        feature_usage = {}
        for i in range(self.n_features):
            feature_usage[i] = sum(self.feature_vars[node_id] == i for node_id in self.feature_vars)
            # Penalize using the same feature multiple times, but with a smaller weight
            regularization += 2 * (feature_usage[i] - 1) * (feature_usage[i] > 1)

        return regularization

    def train(self, X, y):
        """Train the decision tree using CPMpy with class labels"""
        n_features = X.shape[1]
        n_classes = len(np.unique(y))

        if self.max_classes is not None and self.max_classes > n_classes:
            print(f"Warning: max_classes ({self.max_classes}) is greater than number of classes ({n_classes})")
            self.max_classes = n_classes
        
        # Ensure max_depth doesn't exceed the number of useful features
        non_constant_features = 0
        for j in range(X.shape[1]):
            if j in self.categorical_features:
                unique_values = np.unique(X[:, j])
                if len(unique_values) > 1:
                    non_constant_features += 1
            else:
                if X[:, j].min() != X[:, j].max():
                    non_constant_features += 1
        
        if non_constant_features < self.max_depth:
            print(f"Warning: max_depth ({self.max_depth}) is greater than number of non-constant features ({non_constant_features})")
            self.max_depth = max(1, non_constant_features)  # Ensure at least depth 1
            print(f"Reducing max_depth to {self.max_depth}")

        # Try with progressively simpler constraints until we find a solution
        for attempt in range(3):
            try:
                self.create_tree_variables(n_features, n_classes, X)
                total_accuracy = self.add_accuracy_objective(X, y)
                regularization = self.add_regularization_terms()
                
                # Add constraints to force diversity in predictions
                # Ensure we use at least 2 different classes (or max_classes if specified)
                min_classes = min(2, n_classes) if self.max_classes is None else min(self.max_classes, n_classes)
                
                # Create variables to track which classes are used
                class_used = []
                for i in range(n_classes):
                    class_used.append(BoolVar(name=f"class_used_{i}"))
                    class_usage = []
                    for node_id in self.leaf_prediction_vars:
                        class_usage.append(self.leaf_prediction_vars[node_id] == i)
                    self.model += class_used[i] == any(class_usage)
                
                # Ensure at least min_classes are used
                self.model += sum(class_used) >= min_classes
                
                # Maximize accuracy with regularization
                self.model.maximize(100000 * total_accuracy - regularization)
                
                # Set a time limit for solving (in seconds)
                status = self.model.solve(time_limit=300)  # 5 minutes time limit
                print(f"Solver status: {status}")
                
                if status:
                    # Debug: print class distribution
                    class_counts = {}
                    for node_id in self.leaf_prediction_vars:
                        pred = self.leaf_prediction_vars[node_id].value()
                        class_counts[pred] = class_counts.get(pred, 0) + 1
                    print(f"Class distribution in leaves: {class_counts}")
                    
                    return self.extract_tree()
                else:
                    print(f"Attempt {attempt+1} failed, simplifying constraints...")
                    # Simplify for next attempt
                    self.model = Model()  # Reset the model
                    if self.max_depth > 2:
                        self.max_depth -= 1
                        print(f"Reducing max_depth to {self.max_depth}")
            except Exception as e:
                print(f"Error in attempt {attempt+1}: {e}")
                self.model = Model()  # Reset the model
                if self.max_depth > 2:
                    self.max_depth -= 1
                    print(f"Reducing max_depth to {self.max_depth}")
        
        # If all attempts fail, create a simple one-level tree
        print("Creating a simple one-level tree as fallback")
        self.max_depth = 2
        self.model = Model()
        self.create_tree_variables(n_features, n_classes, X)
        total_accuracy = self.add_accuracy_objective(X, y)
        
        # Still enforce class diversity in the fallback solution
        class_used = []
        for i in range(n_classes):
            class_used.append(BoolVar(name=f"class_used_{i}"))
            class_usage = []
            for node_id in self.leaf_prediction_vars:
                class_usage.append(self.leaf_prediction_vars[node_id] == i)
            self.model += class_used[i] == any(class_usage)
        
        # Ensure at least 2 classes are used
        self.model += sum(class_used) >= min(2, n_classes)
        
        self.model.maximize(total_accuracy)
        status = self.model.solve()
        
        if not status:
            raise ValueError("Could not find any solution, even with simplified constraints")
        
        return self.extract_tree()

    def extract_tree(self):
        """Extract the trained decision tree from the solved model"""
        tree = {}
        used_features = set()
        class_distribution = {}
        used_classes = set()

        for node_id in self.feature_vars:
            depth, node = node_id

            selected_feature = self.feature_vars[node_id].value()
            used_features.add(selected_feature)

            tree[node_id] = {"feature": selected_feature}

            # Only get threshold or categorical set for non-leaf nodes
            if depth < self.max_depth - 1:
                if selected_feature in self.categorical_features:
                    # For categorical features, extract the set of values
                    set_vars = self.categorical_set_vars[node_id][selected_feature]
                    selected_values = []
                    for i, var in enumerate(set_vars):
                        if var.value():  # If this value is in the set
                            selected_values.append(
                                self.categorical_values[selected_feature][i]
                            )
                    tree[node_id]["categorical_set"] = selected_values
                else:
                    # For numerical features, extract the threshold
                    threshold = float(self.threshold_vars[node_id].value()) / 100
                    tree[node_id]["threshold"] = threshold

            # For leaf nodes, get prediction
            if depth == self.max_depth - 1:
                prediction = self.leaf_prediction_vars[node_id].value()
                tree[node_id]["prediction"] = prediction
                used_classes.add(prediction)
                class_distribution[prediction] = (
                    class_distribution.get(prediction, 0) + 1
                )

        tree["metadata"] = {
            "used_features": sorted(list(used_features)),
            "n_features_used": len(used_features),
            "used_classes": sorted(list(used_classes)),
            "n_classes_used": len(used_classes),
            "class_distribution": class_distribution,
            "categorical_features": self.categorical_features,
            "no_reuse_features": self.no_reuse_features,
        }

        return tree

    def predict(self, X, tree):
        """Make predictions using the trained tree"""
        predictions = []
        for sample in X:
            node_id = (0, 0)

            while node_id[0] < self.max_depth - 1:
                node = tree[node_id]
                feature_idx = node["feature"]

                if feature_idx in tree["metadata"]["categorical_features"]:
                    # For categorical features, check if the sample's value is in the set
                    if sample[feature_idx] in node["categorical_set"]:
                        node_id = (node_id[0] + 1, node_id[1] * 2)  # Go left
                    else:
                        node_id = (node_id[0] + 1, node_id[1] * 2 + 1)  # Go right
                else:
                    # For numerical features, use threshold comparison
                    if sample[feature_idx] >= node["threshold"]:
                        node_id = (node_id[0] + 1, node_id[1] * 2)  # Go left
                    else:
                        node_id = (node_id[0] + 1, node_id[1] * 2 + 1)  # Go right

            predictions.append(tree[node_id]["prediction"])

        return np.array(predictions)

    def evaluate_cost(self, X, cost_matrix, tree):
        """Evaluate the total cost of predictions using a cost matrix"""
        predictions = self.predict(X, tree)
        total_cost = 0
        for i, pred in enumerate(predictions):
            total_cost += cost_matrix[i, pred]
        return total_cost/X.shape[0]

    def evaluate_accuracy(self, X, y, tree):
        """Evaluate the accuracy of predictions"""
        predictions = self.predict(X, tree)
        correct = (predictions == y).sum()
        return correct / len(y)


# %%


# Usage example
def run_example():
    # Generate sample data with both numerical and categorical features
    # For categorical features, we use integers to represent categories
    X = np.array(
        [
            [1.0, 2.0, 0],  # Categorical feature (index 2) with value 0
            [2.0, 1.0, 1],  # Categorical feature with value 1
            [3.0, 3.0, 2],  # Categorical feature with value 2
            [4.0, 1.0, 0],  # Categorical feature with value 0
            [2.0, 4.0, 1],  # Categorical feature with value 1
            [1.0, 3.0, 2],  # Categorical feature with value 2
        ]
    )

    # Create class labels
    y = np.array([0, 1, 2, 1, 0, 1])

    # Create a cost matrix (n_samples x n_classes) for evaluation
    # Lower values indicate preferred classes for each sample
    cost_matrix = np.array(
        [
            [0.0, 4, 0.5, 1.0, 2.0],  # Sample 0: class 0 has lowest cost
            [1.0, 4, 0.0, 0.5, 2.0],  # Sample 1: class 1 has lowest cost
            [0.5, 4, 1.0, 0.0, 2.0],  # Sample 2: class 2 has lowest cost
            [2.0, 4, 0.0, 1.0, 0.5],  # Sample 3: class 1 has lowest cost
            [5.0, 4, 2.0, 0.5, 1.0],  # Sample 4: class 0 has lowest cost
            [1.0, 4, 0.0, 0.5, 0.5],  # Sample 5: classes 1, 2, 3 have low costs
        ]
    )

    # Specify which features are categorical
    categorical_features = [2]  # The third feature (index 2) is categorical

    # Create trainer with no_reuse_features=True to prevent feature reuse
    trainer = DecisionTreeTrainerCP(
        max_depth=3,
        max_classes=3,
        categorical_features=categorical_features,
        no_reuse_features=True,
    )
    
    # Train using class labels
    tree = trainer.train(X, y)

    print("Trained decision tree:", tree)
    print("\nFeatures used:", tree["metadata"]["used_features"])
    print("Number of features used:", tree["metadata"]["n_features_used"])
    print("Classes used:", tree["metadata"]["used_classes"])
    print("Number of classes used:", tree["metadata"]["n_classes_used"])
    print("Class distribution:", tree["metadata"]["class_distribution"])
    print("Categorical features:", tree["metadata"]["categorical_features"])
    print("No feature reuse:", tree["metadata"]["no_reuse_features"])

    # Evaluate accuracy
    accuracy = trainer.evaluate_accuracy(X, y, tree)
    print("\nAccuracy of predictions:", accuracy)

    # Evaluate cost (for comparison)
    total_cost = trainer.evaluate_cost(X, cost_matrix, tree)
    print("\nTotal cost of predictions:", total_cost)

    # Make predictions
    predictions = trainer.predict(X, tree)
    print("\nPredictions:", predictions)

    # Show accuracy and cost for each prediction
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: true class {y[i]}, predicted class {pred}, correct: {y[i] == pred}, cost {cost_matrix[i, pred]}")


# run_example()

# %%

# Actual usage

random_state = 1234
test_size = 0.40

system = "gcc"
performance_str = "['exec']"
num_configs = 2
performances = json.loads(performance_str.replace("'", '"'))

best_configs = json.load(open("../results/ocs_results.json"))

res = best_configs[system][performance_str][str(num_configs)]
selected_configs = res["selected_configs"]

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

nmdf = (
    perf_matrix_initial[["inputname"] + performances]
    .groupby("inputname", as_index=True)
    # .transform(lambda x: scale(x))
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)
nmdf["worst_case_performance"] = nmdf[performances].max(axis=1)
perf_matrix = pd.merge(
    perf_matrix_initial,
    nmdf,
    suffixes=("_raw", None),
    left_index=True,
    right_index=True,
)
# We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
perf_matrix["worst_case_performance"] = (
    perf_matrix[["inputname", "worst_case_performance"]]
    .groupby("inputname", as_index=True)
    .transform(lambda x: x - x.min())
)

# Split data
train_inp, test_inp = train_test_split(
    perf_matrix["inputname"].unique(),
    test_size=test_size,
    random_state=random_state,
)

# This is only to make our evaluation simpler
train_inp = sorted(train_inp)
test_inp = sorted(test_inp)

train_perf = perf_matrix[perf_matrix.inputname.isin(train_inp)].copy()
test_perf = perf_matrix[perf_matrix.inputname.isin(test_inp)]

features = train_perf[input_features.columns].drop_duplicates()

# Transform features
X = input_preprocessor.fit_transform(features)
feature_names = input_preprocessor.get_feature_names_out()

# Create cost matrix for evaluation
C = (
    perf_matrix[perf_matrix.inputname.isin(train_inp)][
        ["inputname", "configurationID", "worst_case_performance"]
    ]
    .reset_index()
    .pivot(
        index="inputname", columns="configurationID", values="worst_case_performance"
    )
    .sort_values("inputname")
    .reset_index()
    .drop(columns=["inputname"])
    .values
)

# Create class labels (best configuration for each input)
y = np.argmin(C, axis=1)

# Train the model
trainer = DecisionTreeTrainerCP(
    max_depth=2,
    max_classes=5,
    categorical_features=[],
    no_reuse_features=True,
    feature_names=feature_names,
)
tree = trainer.train(X, y)

print("Trained decision tree:", tree)
print("\nFeatures used:", tree["metadata"]["used_features"])
print("Number of features used:", tree["metadata"]["n_features_used"])
print("Classes used:", tree["metadata"]["used_classes"])
print("Number of classes used:", tree["metadata"]["n_classes_used"])
print("Class distribution:", tree["metadata"]["class_distribution"])
print("Categorical features:", tree["metadata"]["categorical_features"])
print("No feature reuse:", tree["metadata"]["no_reuse_features"])

# Evaluate accuracy
accuracy = trainer.evaluate_accuracy(X, y, tree)
print("\nAccuracy of predictions:", accuracy)

# Evaluate cost
total_cost = trainer.evaluate_cost(X, C, tree)
print("\nTotal cost of predictions:", total_cost)

# Make predictions
predictions = trainer.predict(X, tree)
print("\nPredictions:", predictions)

# Show accuracy and cost for each prediction
for i, pred in enumerate(predictions):
    print(f"Sample {i}: true best config {y[i]}, predicted config {pred}, correct: {y[i] == pred}, cost {C[i, pred]}")

# %%
