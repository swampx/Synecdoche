import math
import numpy as np
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear, ExtractIntermediateLayer
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore
import json
from collections import defaultdict
from tensorflow.keras.models import Model
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from rule_remove import _filter_shapelets_with_dual_representation


# ==================== Helper Functions ====================

def recover(seq):
    """Convert sequence to integer list format"""
    res = seq.squeeze()
    res = np.asarray(res, dtype=int).tolist()
    return res



def custom_distance_euclidean_style_anchor(list1, list2):
    """
    Calculate custom Euclidean distance between two lists (with anchor point weighting)
    Matching values (anchor points) reduce the total distance
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    squared_distance = 0
    anchor_points = 0

    for a, b in zip(list1, list2):
        if (a < 1514 < b) or (b < 1514 < a):
            squared_distance += (abs(a - b) + 1000) ** 2
        else:
            squared_distance += (a - b) ** 2

        if a == b:
            anchor_points += 1

    squared_distance /= len(list1)
    distance = np.sqrt(squared_distance)

    # Reduce distance based on number of anchor points (multiply distance by 0.6 for each anchor point)
    while anchor_points > 0:
        distance *= 0.6
        anchor_points -= 1

    return distance


# ==================== Cluster Center Calculation Functions ====================

def calculate_cluster_representations(cluster):
    """
    Calculate multiple representations of a cluster (including range and mean-std)

    Args:
        cluster: cluster data, list of lists, each inner list represents a sequence

    Returns:
        representations: dictionary containing two types of representations
            - 'range': [[min1, max1], [min2, max2], ...]
            - 'mean_std': [[mean1, std1], [mean2, std2], ...]
    """
    cluster_array = np.array(cluster)

    # Calculate statistics for each position
    min_vals = np.min(cluster_array, axis=0)
    max_vals = np.max(cluster_array, axis=0)
    means = np.mean(cluster_array, axis=0)
    stds = np.std(cluster_array, axis=0)
    weights = 1 / (stds + 1)
    # Construct range format (using variance as offset to expand range)
    range_list = []
    for i in range(len(min_vals)):
        offset = 1 * stds[i]
        range_min = int(min_vals[i] - offset)
        range_max = int(max_vals[i] + offset)
        range_list.append([range_min, range_max])

    # Construct mean_std format
    mean_list = []
    weights_list = []
    for i in range(len(means)):
        mean_val = float(means[i])
        weights_val = float(weights[i])
        mean_list.append(mean_val)
        weights_list.append(weights_val)
    mean_std_list = [mean_list, weights_list]

    return {
        'range': range_list,
        'mean_std': mean_std_list
    }


# ==================== Clustering Functions ====================

def clustering(snippets, eps, ratio, class_name):
    """
    Cluster extracted snippets and calculate multiple representations of cluster centers

    Args:
        snippets: list of extracted snippets
        eps: eps parameter for DBSCAN
        ratio: ratio used to calculate min_samples
        class_name: class name

    Returns:
        center_list: list of cluster centers, each element is a dictionary containing two representations:
            {'range': [[min, max], ...], 'mean_std': [[mean, std], ...]}
    """
    # Group sequences by length
    sequences_by_length = defaultdict(list)
    for seq in snippets:
        sequences_by_length[len(seq)].append(seq)

    # Store sequences in each cluster
    clusters = defaultdict(list)
    current_label = 0
    min_samples = max(2, math.ceil(len(snippets) * ratio))

    # Cluster sequences for each length
    for length, seqs in sequences_by_length.items():
        if len(seqs) > 1:  # Cluster only if there are at least two sequences
            # Calculate distance matrix using custom distance function
            distance_matrix = pairwise_distances(
                seqs, metric=custom_distance_euclidean_style_anchor
            )

            # Perform DBSCAN clustering
            clustering_result = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric="precomputed"
            ).fit(distance_matrix)
            cluster_labels = clustering_result.labels_

            # Update cluster numbering to ensure global uniqueness
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Exclude noise points

            for ul in unique_labels:
                # Find all sequences belonging to the current cluster
                seqs_in_cluster = [
                    seqs[i] for i, label in enumerate(cluster_labels) if label == ul
                ]
                clusters[current_label + ul].extend(seqs_in_cluster)

            current_label += len(unique_labels)

    # Convert clusters dictionary to list and calculate two types of representations
    cluster_list = list(clusters.values())
    center_list = []

    for c in cluster_list:
        representations = calculate_cluster_representations(c)
        center_list.append(representations)

    return center_list



# ==================== Snippet Extraction Functions ====================

def find_high_activation_snippets(input_seq, cam, max_length):
    """
    Find snippets with the highest activation values

    Args:
        input_seq: input sequence list
        cam: corresponding class activation map
        max_length: maximum snippet length

    Returns:
        snippets: set of high-activation snippets
    """
    snippets = []

    for i in range(len(input_seq)):
        sequence = recover(input_seq[i])
        activations = cam[i].squeeze()

        # Calculate dynamic threshold
        threshold = np.mean(activations) + 1.5 * np.std(activations)
        j = 0

        while j < len(activations):
            if activations[j] > threshold:
                snippet = []
                while j < len(activations) and activations[j] > threshold:
                    if sequence[j] != 1514:
                        snippet.append((activations[j], sequence[j]))
                    j += 1

                # Process extracted snippets
                snippet_length = len(snippet)
                if 2 <= snippet_length <= max_length:
                    snippets.append(tuple(seq for _, seq in snippet))
                elif snippet_length > max_length:
                    # Find sub-snippet with maximum sum of activation values
                    max_sum = float('-inf')
                    max_start = 0
                    for k in range(snippet_length - max_length):
                        current_sum = sum(
                            snippet[x][0] for x in range(k, k + max_length)
                        )
                        if current_sum > max_sum:
                            max_sum = current_sum
                            max_start = k

                    top_snippet = snippet[max_start:max_start + max_length]
                    snippets.append(tuple(seq for _, seq in top_snippet))
            else:
                j += 1

    return snippets


# ==================== Grad-CAM Functions ====================

def grad_cam(model, input_seq, class_index):
    """
    Calculate class activation map using Grad-CAM (LayerCAM)

    Args:
        model: trained model
        input_seq: input sequence
        class_index: class index

    Returns:
        cam: class activation map
    """
    score = CategoricalScore([class_index])
    replace2linear = ReplaceToLinear()

    # Get embedding layer output
    embedding_layer_output = model.get_layer('tf.compat.v1.squeeze').output
    model_for_embedding_output = Model(
        inputs=model.input,
        outputs=embedding_layer_output
    )
    y = model_for_embedding_output.predict(input_seq)

    # Use LayerCAM to calculate activation map
    cnn_backbone = model.get_layer('model')
    gradcam = Layercam(cnn_backbone, model_modifier=[], clone=False)
    cam = gradcam(score, y, penultimate_layer='conv1d_3')

    return cam





def generate_rules(model, train_seq_datagen, class_name, sample_num, rule_dir,
                   max_length, eps, ratio, score, representation_type='range'):
    """
    Generate classification rules

    Args:
        model: trained model
        train_seq_datagen: training data generator
        class_name: class name
        sample_num: number of samples
        rule_dir: directory to save rules
        max_length: maximum snippet length
        eps: eps parameter for DBSCAN
        ratio: min_samples ratio for DBSCAN
        score: score threshold for filtering
        representation_type: snippet representation type
            - 'range': minimum and maximum range format [min, max] (default)
            - 'mean_std': mean + standard deviation format [mean, std]

    Returns:
        filtered_shapelets: filtered shapelets list (returns corresponding format based on representation_type)
    """
    # Get class index and data
    class_index = train_seq_datagen.class_names.index(class_name)
    input_seq, _ = train_seq_datagen.load_data_from_single_class(
        class_name, sample_num
    )
    if len(input_seq) == 0:
        print('empty class!')
        return
    # Calculate class activation map
    cam = grad_cam(model, input_seq, class_index)

    # Find candidate snippets from positions with high activation values
    snippets = find_high_activation_snippets(input_seq, cam, max_length)

    # Clustering (returns list of dictionaries containing range and mean_std)
    print('Clustering...')
    shapelets = clustering(list(snippets), eps, ratio, class_name)
    print(f"Number of shapelets: {len(shapelets)}")

    # Filtering (use range format for selection, preserve complete dictionary structure)
    print('Filtering...')
    filtered_results = _filter_shapelets_with_dual_representation(
        shapelets,
        train_seq_datagen,
        class_name,
        sample_num_target=-1,
        sample_num_other=1000,
        score=score
    )

    # Select output format based on representation_type
    if representation_type == 'mean_std':
        print('Extracting mean_std format...')
        # Extract mean_std representation and scores
        filtered_shapelets = [
            (shapelet_dict['mean_std'], score_val)
            for shapelet_dict, score_val in filtered_results
        ]
    elif representation_type == 'range':
        # Extract range representation and scores
        filtered_shapelets = [
            (shapelet_dict['range'], score_val)
            for shapelet_dict, score_val in filtered_results
        ]
    else:
        raise ValueError(
            f"Unsupported representation_type: {representation_type}. "
            "Use 'range' or 'mean_std'."
        )

    # Save rules to JSON file
    final_json_path = rule_dir + class_name + '.json'
    with open(final_json_path, 'w') as file:
        json.dump(filtered_shapelets, file, indent=2)

    return filtered_shapelets