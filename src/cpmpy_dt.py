from cpmpy import *
import numpy as np


class DecisionTreeTrainerCP:
    def __init__(self, max_depth=3, max_features=None, max_classes=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_classes = max_classes
        self.model = Model()

    def create_tree_variables(self, n_features, n_classes, X):
        """Create CPMpy variables for the decision tree structure"""
        self.feature_vars = {}
        self.threshold_vars = {}
        self.leaf_prediction_vars = {}
        self.n_classes = n_classes

        # Find min and max values for each feature
        feature_ranges = []
        for j in range(X.shape[1]):
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

                # For non-leaf nodes, ensure threshold is between min and max of selected feature
                if depth < self.max_depth - 1:
                    for feature_idx in range(n_features):
                        feature_selected = self.feature_vars[node_id] == feature_idx
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

    def encode_prediction(self, sample, node_id=(0, 0)):
        """Recursively encode the prediction logic for a sample"""
        depth, node = node_id

        if depth == self.max_depth - 1:
            return self.leaf_prediction_vars[node_id]

        sample_int = [int(x * 100) for x in sample]
        # Use Element constraint instead of direct indexing
        feature_value = Element(sample_int, self.feature_vars[node_id])
        threshold = self.threshold_vars[node_id]
        split_condition = feature_value >= threshold

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

        # Force threshold to be meaningful by ensuring it splits at least one sample differently
        min_val = min(sample_int)
        max_val = max(sample_int)
        self.model += threshold > min_val
        self.model += threshold < max_val

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

            # Only get threshold for non-leaf nodes
            if depth < self.max_depth - 1:
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
        }

        return tree

    def predict(self, X, tree):
        """Make predictions using the trained tree"""
        predictions = []
        for sample in X:
            node_id = (0, 0)

            while node_id[0] < self.max_depth - 1:
                node = tree[node_id]
                if sample[node["feature"]] >= node["threshold"]:
                    node_id = (node_id[0] + 1, node_id[1] * 2)
                else:
                    node_id = (node_id[0] + 1, node_id[1] * 2 + 1)

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
    # Generate sample data
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [3.0, 3.0, 1.0],
            [4.0, 1.0, 4.0],
            [2.0, 4.0, 1.0],
            [1.0, 3.0, 2.0],
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

    trainer = DecisionTreeTrainerCP(max_depth=2, max_classes=2)
    tree = trainer.train(X, cost_matrix)

    print("Trained decision tree:", tree)
    print("\nFeatures used:", tree["metadata"]["used_features"])
    print("Number of features used:", tree["metadata"]["n_features_used"])
    print("Classes used:", tree["metadata"]["used_classes"])
    print("Number of classes used:", tree["metadata"]["n_classes_used"])
    print("Class distribution:", tree["metadata"]["class_distribution"])

    # Evaluate
    total_cost = trainer.evaluate_cost(X, cost_matrix, tree)
    print("\nTotal cost of predictions:", total_cost)

    # Make predictions
    predictions = trainer.predict(X, tree)
    print("\nPredictions:", predictions)

    # Show cost for each prediction
    for i, pred in enumerate(predictions):
        print(f"Sample {i}: predicted class {pred}, cost {cost_matrix[i, pred]}")
