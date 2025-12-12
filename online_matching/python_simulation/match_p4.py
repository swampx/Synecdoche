import os
import json
import numpy as np
import joblib
from pathlib import Path
from data_generator import DataGenerator
from sklearn.metrics import f1_score, precision_recall_fscore_support
import yaml

class PatternClassifier:
    """
    Pattern-based classifier with decision tree fallback classification
    """

    def __init__(self, patterns_dir, decision_tree_model_path=None, padding_value=1514):
        """
        Initialize classifier

        Parameters:
        patterns_dir: Directory containing range pattern JSON files
        decision_tree_model_path: Path to decision tree model file
        padding_value: Padding value used for early termination mechanism
        """
        self.patterns_dir = patterns_dir
        self.padding_value = padding_value
        self.patterns = {}  # {class_name: [(pattern1, priority1), (pattern2, priority2), ...]}
        self.sorted_patterns = []  # [(pattern, priority, class_name), ...] globally sorted by priority
        self.class_names = []

        # Decision tree related
        self.decision_tree_model = None
        self.decision_tree_model_path = decision_tree_model_path

        self.load_patterns()
        self.load_decision_tree()

    def load_patterns(self):
        """
        Load all range patterns from directory
        """
        patterns_path = Path(self.patterns_dir)

        if not patterns_path.exists():
            raise ValueError(f"Pattern directory {self.patterns_dir} does not exist!")

        json_files = list(patterns_path.glob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON files found in directory {self.patterns_dir}!")

        print(f"Loading {len(json_files)} pattern files...")

        all_patterns = []  # Collect all patterns for global sorting

        for json_file in json_files:
            # Extract class name from filename (remove .json extension)
            class_name = json_file.stem

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)

                if class_name not in self.patterns:
                    self.patterns[class_name] = []

                # Parse new format: each element is [pattern, priority]
                for pattern_item in patterns_data:
                    if isinstance(pattern_item, list) and len(pattern_item) == 2:
                        pattern, priority = pattern_item
                        self.patterns[class_name].append((pattern, priority))
                        # Add to global list with class information
                        all_patterns.append((pattern, priority, class_name))
                    else:
                        print(f"  Warning: Skipping pattern with incorrect format: {pattern_item}")

                # Sort within class by priority in descending order (maintain compatibility)
                self.patterns[class_name].sort(key=lambda x: x[1], reverse=True)

                print(f"  Loaded class '{class_name}': {len(self.patterns[class_name])} patterns")

            except Exception as e:
                print(f"  Warning: Error loading file {json_file}: {str(e)}")

        # Global sort by priority in descending order (higher priority first)
        self.sorted_patterns = sorted(all_patterns, key=lambda x: x[1], reverse=True)

        self.class_names = sorted(self.patterns.keys())
        total_patterns = len(self.sorted_patterns)
        print(f"\nTotal loaded {len(self.class_names)} classes, {total_patterns} patterns")
        print(f"Classes: {self.class_names}")

        if total_patterns > 0:
            print(f"Priority range: {self.sorted_patterns[0][1]} (highest) -> {self.sorted_patterns[-1][1]} (lowest)")

    def load_decision_tree(self):
        """
        Load decision tree model
        """
        if self.decision_tree_model_path and os.path.exists(self.decision_tree_model_path):
            try:
                self.decision_tree_model = joblib.load(self.decision_tree_model_path)
                print(f"Loaded decision tree model: {self.decision_tree_model_path}")
            except Exception as e:
                print(f"Warning: Failed to load decision tree model: {str(e)}")
                self.decision_tree_model = None
        else:
            if self.decision_tree_model_path:
                print(f"Warning: Decision tree model file does not exist: {self.decision_tree_model_path}")
            else:
                print("Decision tree model path not provided, will only use pattern matching")

    def check_pattern_match(self, sample, pattern):
        """
        Check if sample can match pattern at some position, with early termination mechanism

        Parameters:
        sample: Sample sequence (1D array)
        pattern: Pattern range list [[x0,y0], [x1,y1], ...]

        Returns:
        tuple: (match_end_position, sample_length)
               match_end_position: start_pos + pattern_length if matched, -1 if not matched
        """
        sample_length = 30
        for i in range(len(sample)):
            if sample[i] == self.padding_value:
                sample_length = i + 1
                break

        pattern_length = len(pattern)
        if pattern_length > sample_length:
            return -1, sample_length

        # Traverse each possible starting position of the sample sequence
        for start_pos in range(sample_length - pattern_length + 1):
            # Early termination: if current position is padding value, stop matching
            if sample[start_pos] == self.padding_value:
                break

            match = True
            # Check if each dimension of pattern is within range at this position
            for i, (x_min, x_max) in enumerate(pattern):
                sample_value = sample[start_pos + i]

                # If padding value is encountered during matching, consider it as failed match
                if sample_value == self.padding_value:
                    match = False
                    break

                # Check if sample value is within range
                if not (x_min <= sample_value <= x_max):
                    match = False
                    break

            if match:
                return start_pos + pattern_length, sample_length

        return -1, sample_length

    def predict_sample(self, sample):
        """
        Predict class for single sample, first try pattern matching, use decision tree if failed

        Parameters:
        sample: Sample sequence (1D array)

        Returns:
        tuple: (predicted_class, confidence_score, match_pos, method)
               method: 'pattern' or 'decision_tree'
        """
        # First try pattern matching
        for pattern, priority, class_name in self.sorted_patterns:
            match_pos, sample_length = self.check_pattern_match(sample, pattern)
            if match_pos != -1:
                # Found first matching pattern, return that class immediately
                return class_name, 1.0, match_pos, 'pattern'

        # If pattern matching fails, try using decision tree
        if self.decision_tree_model is not None:
            try:
                # Flatten sample for decision tree prediction
                sample_flattened = sample.reshape(1, -1)

                # Decision tree prediction
                tree_pred = self.decision_tree_model.predict(sample_flattened[:, :4])[0]

                # Get prediction probability (if supported)
                confidence = 1.0
                if hasattr(self.decision_tree_model, 'predict_proba'):
                    proba = self.decision_tree_model.predict_proba(sample_flattened[:, :4])[0]
                    confidence = np.max(proba)

                # Convert numeric prediction to class name
                if isinstance(tree_pred, (int, np.integer)):
                    if tree_pred < len(self.class_names):
                        predicted_class = self.class_names[tree_pred]
                    else:
                        predicted_class = f"class_{tree_pred}"
                else:
                    predicted_class = str(tree_pred)

                # Set match position to sample length
                match_pos = sample_length

                return predicted_class, confidence, match_pos, 'decision_tree'

            except Exception as e:
                print(f"Decision tree prediction error: {str(e)}")

        # If all failed, return no match
        return None, 0.0, -1, 'none'

    def predict_batch(self, samples):
        """
        Predict class for batch of samples

        Parameters:
        samples: List of samples

        Returns:
        list: [(predicted_class, confidence, match_pos, method), ...]
        """
        predictions = []
        for sample in samples:
            pred_class, confidence, match_pos, method = self.predict_sample(sample)
            predictions.append((pred_class, confidence, match_pos, method))

        return predictions

    def calculate_f1_scores(self, true_labels, pred_labels, class_names):
        """
        Calculate F1 scores

        Parameters:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: List of class names

        Returns:
        dict: Dictionary containing various F1 scores
        """
        # Filter out None predictions (unmatched samples)
        valid_indices = [i for i, pred in enumerate(pred_labels) if pred is not None]

        if not valid_indices:
            return {
                'macro_f1': 0.0,
                'micro_f1': 0.0,
                'weighted_f1': 0.0,
                'class_f1': {class_name: 0.0 for class_name in class_names}
            }

        valid_true = [true_labels[i] for i in valid_indices]
        valid_pred = [pred_labels[i] for i in valid_indices]

        # Calculate various F1 scores
        macro_f1 = f1_score(valid_true, valid_pred, labels=class_names, average='macro', zero_division=0)
        micro_f1 = f1_score(valid_true, valid_pred, labels=class_names, average='micro', zero_division=0)
        weighted_f1 = f1_score(valid_true, valid_pred, labels=class_names, average='weighted', zero_division=0)

        # Calculate F1 score for each class
        precision, recall, class_f1_scores, support = precision_recall_fscore_support(
            valid_true, valid_pred, labels=class_names, zero_division=0
        )

        class_f1 = {class_names[i]: class_f1_scores[i] for i in range(len(class_names))}

        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'class_f1': class_f1
        }


    def evaluate_datagenerator(self, data_generator, verbose=True):
        """
        Evaluate all data in DataGenerator

        Parameters:
        data_generator: DataGenerator instance
        verbose: Whether to display detailed information

        Returns:
        dict: Evaluation results
        """
        print("Starting evaluation...")

        total_samples = 0
        correct_predictions = 0
        pattern_matched = 0  # Number of samples successfully matched by patterns
        tree_used = 0  # Number of samples classified by decision tree
        pattern_correct = 0  # Number of correctly matched patterns
        tree_correct = 0  # Number of correctly classified by decision tree

        # Class name mapping
        generator_classes = data_generator.class_names

        # Store all predictions for F1 calculation
        all_true_labels = []
        all_pred_labels = []
        pattern_true_labels = []
        pattern_pred_labels = []

        # Store match position information
        all_match_positions = []  # Store all match positions
        pattern_match_positions = []  # Pattern matching positions
        tree_match_positions = []  # Decision tree classification positions
        class_match_positions = {class_name: [] for class_name in generator_classes}

        # New: Match position statistics dictionary {position: sample_count}
        match_positions_dict = {}  # Contains all match positions
        pattern_match_positions_dict = {}  # Only contains pattern match positions

        # Store detailed results
        results = {
            'total_samples': 0,
            'correct_predictions': 0,
            'pattern_matched': 0,
            'tree_used': 0,
            'pattern_correct': 0,
            'tree_correct': 0,
            'accuracy': 0.0,
            'pattern_match_rate': 0.0,
            'tree_usage_rate': 0.0,
            'tree_accuracy': 0.0,
            'pattern_accuracy': 0.0,
            'class_results': {class_name: {
                'total': 0, 'correct': 0,
                'pattern_matched': 0, 'tree_used': 0,
                'match_positions': []
            } for class_name in generator_classes},
            'confusion_matrix': {},
            'match_position_stats': {
                'overall_avg_match_pos': 0.0,
                'pattern_avg_match_pos': 0.0,
                'tree_avg_match_pos': 0.0,
                'overall_match_positions': [],
                'pattern_match_positions': [],
                'tree_match_positions': [],
                'class_avg_match_pos': {},
                'match_positions_dict': {},  # All match position statistics
                'pattern_match_positions_dict': {}  # New: Only pattern match position statistics
            },
            'f1_scores': {},  # New: F1 score field
            'pattern_f1_scores': {}  # New: Pattern matching F1 score field
        }

        # Iterate through all batches
        for batch_idx in range(len(data_generator)):
            if verbose and batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(data_generator)}")

            batch_data, batch_labels = data_generator[batch_idx]

            # Process each sample in batch
            for sample_idx, (sample, label) in enumerate(zip(batch_data, batch_labels)):
                # Remove extra dimensions (if any)
                if sample.ndim > 1:
                    sample = sample.squeeze()

                # Get true class
                true_class_idx = np.argmax(label)
                true_class = generator_classes[true_class_idx]

                # Predict (will always have result now)
                pred_class, confidence, match_pos, method = self.predict_sample(sample)

                total_samples += 1
                results['class_results'][true_class]['total'] += 1

                # Add to label lists for F1 calculation
                all_true_labels.append(true_class)
                all_pred_labels.append(pred_class)

                # Record match position (all samples will have position)
                if match_pos > 0:  # Only record valid match positions
                    all_match_positions.append(match_pos)
                    class_match_positions[true_class].append(match_pos)
                    results['class_results'][true_class]['match_positions'].append(match_pos)

                    # Update all match position statistics dictionary
                    if match_pos not in match_positions_dict:
                        match_positions_dict[match_pos] = 0
                    match_positions_dict[match_pos] += 1

                # Classify statistics by method
                if method == 'pattern':
                    pattern_matched += 1
                    if match_pos > 0:
                        pattern_match_positions.append(match_pos)
                        # Update pattern match position statistics dictionary
                        if match_pos not in pattern_match_positions_dict:
                            pattern_match_positions_dict[match_pos] = 0
                        pattern_match_positions_dict[match_pos] += 1
                    results['class_results'][true_class]['pattern_matched'] += 1

                    # Add to label lists for pattern matching F1 calculation
                    pattern_true_labels.append(true_class)
                    pattern_pred_labels.append(pred_class)

                    # Check accuracy of pattern matching
                    if pred_class == true_class:
                        pattern_correct += 1

                elif method == 'decision_tree':
                    tree_used += 1
                    if match_pos > 0:
                        tree_match_positions.append(match_pos)
                    results['class_results'][true_class]['tree_used'] += 1

                    # Check accuracy of decision tree classification
                    if pred_class == true_class:
                        tree_correct += 1

                # Check overall prediction correctness
                if pred_class == true_class:
                    correct_predictions += 1
                    results['class_results'][true_class]['correct'] += 1

                # Update confusion matrix
                if true_class not in results['confusion_matrix']:
                    results['confusion_matrix'][true_class] = {}

                if pred_class not in results['confusion_matrix'][true_class]:
                    results['confusion_matrix'][true_class][pred_class] = 0
                results['confusion_matrix'][true_class][pred_class] += 1

        # Calculate final results
        results['total_samples'] = total_samples
        results['correct_predictions'] = correct_predictions
        results['pattern_matched'] = pattern_matched
        results['tree_used'] = tree_used
        results['pattern_correct'] = pattern_correct
        results['tree_correct'] = tree_correct

        results['pattern_match_rate'] = pattern_matched / total_samples if total_samples > 0 else 0
        results['tree_usage_rate'] = tree_used / total_samples if total_samples > 0 else 0
        results['accuracy'] = correct_predictions / total_samples if total_samples > 0 else 0
        results['pattern_accuracy'] = pattern_correct / pattern_matched if pattern_matched > 0 else 0
        results['tree_accuracy'] = tree_correct / tree_used if tree_used > 0 else 0

        # Calculate F1 scores
        results['f1_scores'] = self.calculate_f1_scores(all_true_labels, all_pred_labels, generator_classes)
        results['pattern_f1_scores'] = self.calculate_f1_scores(pattern_true_labels, pattern_pred_labels,
                                                                generator_classes)

        # Calculate match position statistics
        results['match_position_stats']['overall_match_positions'] = all_match_positions
        results['match_position_stats']['pattern_match_positions'] = pattern_match_positions
        results['match_position_stats']['tree_match_positions'] = tree_match_positions
        results['match_position_stats']['match_positions_dict'] = match_positions_dict
        results['match_position_stats']['pattern_match_positions_dict'] = pattern_match_positions_dict

        if all_match_positions:
            results['match_position_stats']['overall_avg_match_pos'] = np.mean(all_match_positions)
        else:
            results['match_position_stats']['overall_avg_match_pos'] = 0.0

        if pattern_match_positions:
            results['match_position_stats']['pattern_avg_match_pos'] = np.mean(pattern_match_positions)
        else:
            results['match_position_stats']['pattern_avg_match_pos'] = 0.0

        if tree_match_positions:
            results['match_position_stats']['tree_avg_match_pos'] = np.mean(tree_match_positions)
        else:
            results['match_position_stats']['tree_avg_match_pos'] = 0.0

        # Calculate average match position for each class
        for class_name in generator_classes:
            class_positions = class_match_positions[class_name]
            if class_positions:
                results['match_position_stats']['class_avg_match_pos'][class_name] = np.mean(class_positions)
            else:
                results['match_position_stats']['class_avg_match_pos'][class_name] = 0.0

        if verbose:
            self.print_evaluation_results(results)

        return results

    def print_evaluation_results(self, results):
        """
        Print evaluation results
        """
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)

        print(f"Total samples: {results['total_samples']}")
        print(f"  - Pattern matched: {results['pattern_matched']}")
        print(f"  - Decision tree matched: {results['tree_used']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print()
        print(f"Pattern match rate: {results['pattern_match_rate']:.4f}")
        print(f"Decision tree match rate: {results['tree_usage_rate']:.4f}")
        print()
        print(f"Overall accuracy: {results['accuracy']:.4f}")
        print(f"Pattern accuracy: {results['pattern_accuracy']:.4f}")
        print(f"Decision tree accuracy: {results['tree_accuracy']:.4f}")

        # Print F1 scores
        print("\n" + "=" * 50)
        print("F1 Score Statistics")
        print("=" * 50)

        print("Overall system F1 scores:")
        print(f"  Macro-average F1: {results['f1_scores']['macro_f1']:.4f}")
        print(f"  Micro-average F1: {results['f1_scores']['micro_f1']:.4f}")
        print(f"  Weighted-average F1: {results['f1_scores']['weighted_f1']:.4f}")

        print("\nPattern matching F1 scores:")
        print(f"  Macro-average F1: {results['pattern_f1_scores']['macro_f1']:.4f}")
        print(f"  Micro-average F1: {results['pattern_f1_scores']['micro_f1']:.4f}")
        print(f"  Weighted-average F1: {results['pattern_f1_scores']['weighted_f1']:.4f}")

        print("\nF1 scores by class:")
        print(f"{'Class':<15} {'Overall F1':<10} {'Pattern F1':<10}")
        print("-" * 35)
        for class_name in results['class_results'].keys():
            overall_f1 = results['f1_scores']['class_f1'][class_name]
            pattern_f1 = results['pattern_f1_scores']['class_f1'][class_name]
            print(f"{class_name:<15} {overall_f1:<10.4f} {pattern_f1:<10.4f}")

        # Print match position statistics
        print("\n" + "=" * 50)
        print("Match Position Statistics")
        print("=" * 50)
        overall_avg_pos = results['match_position_stats']['overall_avg_match_pos']
        pattern_avg_pos = results['match_position_stats']['pattern_avg_match_pos']
        tree_avg_pos = results['match_position_stats']['tree_avg_match_pos']

        print(f"Overall average match position: {overall_avg_pos:.2f}")
        print(f"Pattern match average position: {pattern_avg_pos:.2f}")
        print(f"Decision tree match average position: {tree_avg_pos:.2f}")

        print(f"\nAverage match position by class:")
        print("-" * 50)
        print(f"{'Class':<15} {'Avg Position':<12} {'Pattern Match':<10} {'Tree Match':<10}")
        print("-" * 50)

        for class_name in results['class_results'].keys():
            avg_pos = results['match_position_stats']['class_avg_match_pos'][class_name]
            pattern_matched = results['class_results'][class_name]['pattern_matched']
            tree_matched = results['class_results'][class_name]['tree_used']
            print(f"{class_name:<15} {avg_pos:<12.2f} {pattern_matched:<10} {tree_matched:<10}")

        print("\nDetailed results by class:")
        print("-" * 80)
        print(f"{'Class':<15} {'Total':<6} {'Correct':<8} {'Accuracy':<8}")
        print("-" * 80)

        for class_name, class_result in results['class_results'].items():
            total = class_result['total']
            correct = class_result['correct']
            accuracy = correct / total if total > 0 else 0
            print(f"{class_name:<15} {total:<6} {correct:<8} {accuracy:<8.4f}")

        print("\nConfusion matrix:")
        print("-" * 40)
        for true_class, predictions in results['confusion_matrix'].items():
            print(f"{true_class}:")
            for pred_class, count in predictions.items():
                print(f"  -> {pred_class}: {count}")

        # New: Print pattern match position statistics separately
        print("\n" + "=" * 60)
        print("Pattern Match Position Detailed Statistics")
        print("=" * 60)
        pattern_match_positions_dict = results['match_position_stats']['pattern_match_positions_dict']

        if pattern_match_positions_dict:
            total_pattern_samples = sum(pattern_match_positions_dict.values())
            print(f"Total samples successfully matched by patterns: {total_pattern_samples}")
            print(f"Average pattern match position: {pattern_avg_pos:.2f}")
            print("\nPattern match position distribution:")
            print(f"{'Position':<8} {'Sample Count':<10} {'Percentage':<10}")
            print("-" * 30)

            # Sort and print by position
            for pos in sorted(pattern_match_positions_dict.keys()):
                count = pattern_match_positions_dict[pos]
                percentage = (count / total_pattern_samples) * 100
                print(f"{pos:<8} {count:<10} {percentage:<10.2f}%")

            # Calculate some key statistics
            positions = sorted(pattern_match_positions_dict.keys())
            counts = [pattern_match_positions_dict[pos] for pos in positions]
            cumulative_counts = np.cumsum(counts)
            cumulative_ratios = cumulative_counts / total_pattern_samples

            # Find 50% and 90% percentiles
            median_idx = np.searchsorted(cumulative_ratios, 0.5)
            percentile_90_idx = np.searchsorted(cumulative_ratios, 0.9)

            median_pos = positions[median_idx] if median_idx < len(positions) else positions[-1]
            percentile_90_pos = positions[percentile_90_idx] if percentile_90_idx < len(positions) else positions[-1]

            print(f"\nKey statistics for pattern matching:")
            print(f"  Earliest match position: {min(positions)}")
            print(f"  Latest match position: {max(positions)}")
            print(f"  50% samples match position not exceeding: {median_pos}")
            print(f"  90% samples match position not exceeding: {percentile_90_pos}")
        else:
            print("No samples successfully matched by patterns")


        match_positions_dict = results['match_position_stats']['pattern_match_positions_dict']



def main():
    """
    Main function - example usage
    """
    print("Pattern-based classifier + Decision tree fallback")
    print("=" * 60)

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = config['dataset']['name']

    # Configure paths
    patterns_dir = config['rules_generating']['rules_path'] + dataset
    test_data_dir = config['dataset']['txt_folder']+ dataset + '_t/'
    decision_tree_model_path = config['cnn_training']['model_saved_path'] + f"{dataset}/decision_trees/best_model.pkl"
    num_classes = config['dataset']['num_classes']

    # Create classifier
    classifier = PatternClassifier(
        patterns_dir=patterns_dir,
        decision_tree_model_path=decision_tree_model_path,
        padding_value=1514
    )

    # Create data generator
    data_generator = DataGenerator(
        data_dir=test_data_dir,
        num_classes=num_classes)

    # Perform evaluation
    results = classifier.evaluate_datagenerator(data_generator)


if __name__ == "__main__":
    main()