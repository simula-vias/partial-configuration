from cpmpy import *
import numpy as np


class DecisionTreeTrainerCP:
    def __init__(self, max_depth=3, max_features=None, max_classes=None, categorical_features=None, no_reuse_features=True):
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_classes = max_classes
        self.categorical_features = categorical_features or []  # List of indices of categorical features
        self.no_reuse_features = no_reuse_features  # Whether to prevent feature reuse in paths
        self.model = Model()

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
        for j in range(X.shape[1]):
            if j in self.categorical_features:
                unique_values = np.unique(X[:, j])
                self.categorical_values[j] = unique_values
                # For categorical features, we still need a range for the threshold variables
                # (these won't be used directly but are needed for the model structure)
                feature_ranges.append((0, len(unique_values) - 1))
            else:
                feature_ranges.append((X[:, j].min(), X[:, j].max()))

        # Create variables for each node in the tree
        for depth in range(self.max_depth):
            for node in range(2**depth):
                node_id = (depth, node)

                # Feature selection variables
                self.feature_vars[node_id] = IntVar(
                    0, n_features - 1, name=f"feature_{depth}_{node}"
                )

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
                        self.model += self.feature_vars[node_i] != self.feature_vars[node_j]

    def encode_prediction(self, sample, node_id=(0, 0)):
        """Recursively encode the prediction logic for a sample"""
        depth, node = node_id

        if depth == self.max_depth - 1:
            return self.leaf_prediction_vars[node_id]

        # Create a variable for the split condition
        split_condition = BoolVar(name=f"split_{depth}_{node}")
        
        # Get the feature index for this node
        feature_var = self.feature_vars[node_id]
        
        # Handle both categorical and numerical features
        categorical_conditions = []
        numerical_conditions = []
        
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
                categorical_conditions.append(feature_selected & in_set)
            else:
                # For numerical features, use the threshold comparison
                sample_value = int(sample[feat_idx] * 100)
                threshold = self.threshold_vars[node_id]
                numerical_conditions.append(feature_selected & (sample_value >= threshold))
        
        # The split condition is true if any of the feature-specific conditions are true
        if categorical_conditions and numerical_conditions:
            self.model += split_condition == any(categorical_conditions + numerical_conditions)
        elif categorical_conditions:
            self.model += split_condition == any(categorical_conditions)
        else:
            self.model += split_condition == any(numerical_conditions)

        left_child = (depth + 1, node * 2)
        right_child = (depth + 1, node * 2 + 1)

        # Create variables for the left and right predictions
        left_pred = self.encode_prediction(sample, left_child)
        right_pred = self.encode_prediction(sample, right_child)

        # Create a new variable for the result
        result = IntVar(0, self.n_classes - 1)

        # Add constraints to enforce the decision logic
        self.model += (result == left_pred) | (
            result == right_pred
        )  # Result must be one of the two options

        # Use direct logical constraints instead of implications
        self.model += (~split_condition) | (
            result == left_pred
        )  # If condition is true, use left
        self.model += (split_condition) | (
            result == right_pred
        )  # If condition is false, use right

        return result

    def add_cost_objective(self, X, cost_matrix):
        """Add cost objective for the training data using a cost matrix"""
        # Create variables to track the cost for each sample
        sample_costs = []

        for i, sample in enumerate(X):
            prediction = self.encode_prediction(sample)

            # Create a variable for the cost of this sample
            sample_cost = IntVar(0, int(np.max(cost_matrix) * 100), name=f"cost_{i}")

            # For each possible class, add a constraint that if this class is predicted,
            # the sample cost equals the corresponding cost from the cost matrix
            cost_constraints = []
            for class_idx in range(self.n_classes):
                # Scale costs by 100 to work with integer variables
                scaled_cost = int(cost_matrix[i, class_idx] * 100)
                cost_constraints.append(
                    ((prediction == class_idx) & (sample_cost == scaled_cost))
                )

            # One of these constraints must be true
            self.model += any(cost_constraints)

            sample_costs.append(sample_cost)

        # Total cost is the sum of all sample costs
        total_cost = sum(sample_costs)

        return total_cost

    def add_regularization_terms(self):
        """Create regularization terms for the objective function"""
        regularization = 0

        # Add feature selection terms to objective
        for node_id in self.feature_vars:
            depth, _ = node_id
            weight = 2 ** (self.max_depth - depth - 1)
            regularization += weight * self.feature_vars[node_id]

        return regularization

    def train(self, X, cost_matrix):
        """Train the decision tree using CPMpy with a cost matrix

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            cost_matrix: Cost matrix of shape (n_samples, n_classes)
                         where cost_matrix[i,j] is the cost of predicting class j for sample i
        """
        n_features = X.shape[1]
        n_classes = cost_matrix.shape[1]

        if self.max_classes is not None and self.max_classes > n_classes:
            print(
                f"Warning: max_classes ({self.max_classes}) is greater than number of classes ({n_classes})"
            )
            self.max_classes = n_classes

        self.create_tree_variables(n_features, n_classes, X)

        # Add cost objective
        total_cost = self.add_cost_objective(X, cost_matrix)

        # Add regularization terms
        regularization = self.add_regularization_terms()

        # Set minimization objective: cost + regularization
        # Scale regularization to be a small fraction of the cost
        self.model.minimize(total_cost + regularization)

        # Solve and check if solution was found
        status = self.model.solve()
        print(f"Solver status: {status}")

        # Debug: print some variable values
        for node_id in self.feature_vars:
            print(f"\nNode {node_id}:")
            print(f"Feature var value: {self.feature_vars[node_id].value()}")
            print(f"Threshold var value: {self.threshold_vars[node_id].value()}")
            if node_id[0] == self.max_depth - 1:  # leaf node
                print(
                    f"Leaf prediction value: {self.leaf_prediction_vars[node_id].value()}"
                )
            # Print categorical set values if applicable
            if self.categorical_features and node_id in self.categorical_set_vars:
                for feat_idx in self.categorical_set_vars[node_id]:
                    set_values = [var.value() for var in self.categorical_set_vars[node_id][feat_idx]]
                    print(f"Categorical set for feature {feat_idx}: {set_values}")

        if not status:
            raise ValueError(
                "No solution found - try increasing max_depth, max_features, or max_classes"
            )

        # Only extract tree if we found a solution
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
                            selected_values.append(self.categorical_values[selected_feature][i])
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
        """Evaluate the total cost of predictions"""
        predictions = self.predict(X, tree)
        total_cost = 0
        for i, pred in enumerate(predictions):
            total_cost += cost_matrix[i, pred]
        return total_cost


# Example usage
if __name__ == "__main__":
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

    # Create a cost matrix (n_samples x n_classes)
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
    trainer = DecisionTreeTrainerCP(max_depth=3, max_classes=3, categorical_features=categorical_features, no_reuse_features=True)
    tree = trainer.train(X, cost_matrix)

    print("Trained decision tree:", tree)
    print("\nFeatures used:", tree["metadata"]["used_features"])
    print("Number of features used:", tree["metadata"]["n_features_used"])
    print("Classes used:", tree["metadata"]["used_classes"])
    print("Number of classes used:", tree["metadata"]["n_classes_used"])
    print("Class distribution:", tree["metadata"]["class_distribution"])
    print("Categorical features:", tree["metadata"]["categorical_features"])
    print("No feature reuse:", tree["metadata"]["no_reuse_features"])

    # Evaluate
    total_cost = trainer.evaluate_cost(X, cost_matrix, tree)
    print("\nTotal cost of predictions:", total_cost)

    # Make predictions
    predictions = trainer.predict(X, tree)
    print("\nPredictions:", predictions)

    # Show cost for each prediction
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: predicted class {pred}, cost {cost_matrix[i, pred]}")
