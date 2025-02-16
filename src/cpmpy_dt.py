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
                self.feature_vars[node_id] = IntVar(0, n_features - 1, 
                                                  name=f"feature_{depth}_{node}")
                
                # Threshold variable - allow any value between min and max
                min_val = min(r[0] for r in feature_ranges) * 100
                max_val = max(r[1] for r in feature_ranges) * 100
                self.threshold_vars[node_id] = IntVar(int(min_val), int(max_val),
                                                    name=f"threshold_{depth}_{node}")
                
                # For non-leaf nodes, ensure threshold is between min and max of selected feature
                if depth < self.max_depth - 1:
                    for feature_idx in range(n_features):
                        feature_selected = (self.feature_vars[node_id] == feature_idx)
                        feature_min = int(feature_ranges[feature_idx][0] * 100)
                        feature_max = int(feature_ranges[feature_idx][1] * 100)
                        self.model += ~feature_selected | (self.threshold_vars[node_id] >= feature_min)
                        self.model += ~feature_selected | (self.threshold_vars[node_id] <= feature_max)
                
                # For leaf nodes, create prediction variables for each class
                if depth == self.max_depth - 1:
                    self.leaf_prediction_vars[node_id] = IntVar(0, n_classes - 1,
                                                              name=f"leaf_{depth}_{node}")
        
        # Feature usage tracking
        if self.max_features is not None:
            self.feature_used = []
            for i in range(n_features):
                self.feature_used.append(BoolVar(name=f"feature_used_{i}"))
                
                feature_usage = []
                for node_id in self.feature_vars:
                    feature_usage.append(self.feature_vars[node_id] == i)
                self.model += (self.feature_used[i] == any(feature_usage))
            
            self.model += (sum(self.feature_used) <= self.max_features)
            
        # Class usage tracking
        if self.max_classes is not None:
            self.class_used = []
            for i in range(n_classes):
                self.class_used.append(BoolVar(name=f"class_used_{i}"))
                
                class_usage = []
                for node_id in self.leaf_prediction_vars:
                    class_usage.append(self.leaf_prediction_vars[node_id] == i)
                self.model += (self.class_used[i] == any(class_usage))
            
            self.model += (sum(self.class_used) <= self.max_classes)
            
            # Ensure used classes are consecutive
            for i in range(1, n_classes):
                self.model += (self.class_used[i] <= self.class_used[i-1])
    
    def encode_prediction(self, sample, node_id=(0, 0)):
        """Recursively encode the prediction logic for a sample"""
        depth, node = node_id
        
        if depth == self.max_depth - 1:
            return self.leaf_prediction_vars[node_id]
        
        sample_int = [int(x * 100) for x in sample]
        # Use Element constraint instead of direct indexing
        feature_value = Element(sample_int, self.feature_vars[node_id])
        threshold = self.threshold_vars[node_id]
        split_condition = (feature_value >= threshold)
        
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
        self.model += (~split_condition) | (result == left_pred)      # If condition is true, use left
        self.model += (split_condition) | (result == right_pred)      # If condition is false, use right
        
        # Force threshold to be meaningful by ensuring it splits at least one sample differently
        min_val = min(sample_int)
        max_val = max(sample_int)
        self.model += (threshold > min_val)
        self.model += (threshold < max_val)
        
        return result
    
    def add_training_constraints(self, X, y_sets):
        """Add constraints for the training data with multiple valid classes per sample"""
        for sample, valid_classes in zip(X, y_sets):
            prediction = self.encode_prediction(sample)
            # Create constraint that prediction must be one of the valid classes
            self.model += (any([prediction == class_label for class_label in valid_classes]))
    
    def add_regularization_constraints(self):
        """Add constraints to prefer simpler trees"""
        # Create a single objective function
        objective = 0
        
        # Add feature selection terms to objective
        for node_id in self.feature_vars:
            depth, _ = node_id
            weight = 2 ** (self.max_depth - depth - 1)
            objective += weight * self.feature_vars[node_id]
        
        # Add class usage terms to objective if needed
        if self.max_classes is not None:
            for class_idx in range(self.n_classes):
                objective += 10 * class_idx * self.class_used[class_idx]
        
        # Set single minimization objective
        self.model.minimize(objective)
    
    def train(self, X, y_sets):
        """Train the decision tree using CPMpy with multi-label data"""
        n_features = X.shape[1]
        # Get unique classes from all sets
        all_classes = set()
        for class_set in y_sets:
            all_classes.update(class_set)
        n_classes = len(all_classes)
        
        if self.max_classes is not None and self.max_classes > n_classes:
            print(f"Warning: max_classes ({self.max_classes}) is greater than number of unique classes ({n_classes})")
            self.max_classes = n_classes
        
        self.create_tree_variables(n_features, n_classes, X)
        self.add_training_constraints(X, y_sets)
        self.add_regularization_constraints()
        
        # Solve and check if solution was found
        status = self.model.solve()
        print(f"Solver status: {status}")
        
        # Debug: print some variable values
        for node_id in self.feature_vars:
            print(f"\nNode {node_id}:")
            print(f"Feature var value: {self.feature_vars[node_id].value()}")
            print(f"Threshold var value: {self.threshold_vars[node_id].value()}")
            if node_id[0] == self.max_depth - 1:  # leaf node
                print(f"Leaf prediction value: {self.leaf_prediction_vars[node_id].value()}")
        
        if not status:
            raise ValueError("No solution found - try increasing max_depth, max_features, or max_classes")
        
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
            
            tree[node_id] = {
                'feature': selected_feature
            }
            
            # Only get threshold for non-leaf nodes
            if depth < self.max_depth - 1:
                threshold = float(self.threshold_vars[node_id].value()) / 100
                tree[node_id]['threshold'] = threshold
            
            # For leaf nodes, get prediction
            if depth == self.max_depth - 1:
                prediction = self.leaf_prediction_vars[node_id].value()
                tree[node_id]['prediction'] = prediction
                used_classes.add(prediction)
                class_distribution[prediction] = class_distribution.get(prediction, 0) + 1
        
        tree['metadata'] = {
            'used_features': sorted(list(used_features)),
            'n_features_used': len(used_features),
            'used_classes': sorted(list(used_classes)),
            'n_classes_used': len(used_classes),
            'class_distribution': class_distribution
        }
        
        return tree

    def predict(self, X, tree):
        """Make predictions using the trained tree"""
        predictions = []
        for sample in X:
            node_id = (0, 0)
            
            while node_id[0] < self.max_depth - 1:
                node = tree[node_id]
                if sample[node['feature']] >= node['threshold']:
                    node_id = (node_id[0] + 1, node_id[1] * 2)
                else:
                    node_id = (node_id[0] + 1, node_id[1] * 2 + 1)
            
            predictions.append(tree[node_id]['prediction'])
        
        return np.array(predictions)
    
    def evaluate(self, X, y_sets, tree):
        """Evaluate predictions against multi-label data"""
        predictions = self.predict(X, tree)
        correct = 0
        for pred, valid_classes in zip(predictions, y_sets):
            if pred in valid_classes:
                correct += 1
        return correct / len(X)

# Example usage
if __name__ == "__main__":
    # Generate sample multi-label data
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [3.0, 3.0, 1.0],
        [4.0, 1.0, 4.0],
        [2.0, 4.0, 1.0],
        [1.0, 3.0, 2.0]
    ])
    # Each sample can belong to multiple classes
    y_sets = [
        {0, 1},     # Sample 0 can be either class 0 or 1
        {1, 2},     # Sample 1 can be either class 1 or 2
        {0, 2},     # Sample 2 can be either class 0 or 2
        {1, 3},     # Sample 3 can be either class 1 or 3
        {0, 2},     # Sample 4 can be either class 0 or 2
        {1, 2, 3}   # Sample 5 can be class 1, 2, or 3
    ]
    
    trainer = DecisionTreeTrainerCP(max_depth=3, max_classes=2)
    tree = trainer.train(X, y_sets)
    
    print("Trained decision tree:", tree)
    print("\nFeatures used:", tree['metadata']['used_features'])
    print("Number of features used:", tree['metadata']['n_features_used'])
    print("Classes used:", tree['metadata']['used_classes'])
    print("Number of classes used:", tree['metadata']['n_classes_used'])
    print("Class distribution:", tree['metadata']['class_distribution'])
    
    # Evaluate
    accuracy = trainer.evaluate(X, y_sets, tree)
    print("\nAccuracy (prediction matches any valid class):", accuracy)