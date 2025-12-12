import numpy as np

def euclidean_distance_between_ts(a, b, weight):
    """
    Calculate the minimum Euclidean distance between time series and shapelet.
    """
    a = np.array(a).squeeze()
    b = np.array(b).squeeze()
    min_distance = float('inf')  # Initialize minimum distance as infinity
    len_a = len(a)
    len_b = len(b)
    match_point = 0

    if len_a == len_b:
        min_distance = np.linalg.norm(weight * (a - b))

    elif len_a < len_b:  # Traverse all possible positions on the time series
        for start in range(len_b - len_a + 1):
            end = start + len_a
            distance = np.linalg.norm(weight * (b[start:end] - a)) / len_a
            if distance < min_distance:
                min_distance = distance
                match_point = end
    else:
        for start in range(len_a - len_b + 1):
            end = start + len_b
            distance = np.linalg.norm(weight * (a[start:end] - b)) / len_b
            if distance < min_distance:
                min_distance = distance
                match_point = end

    return min_distance, match_point


def check_range_shapelet_match(sample, range_shapelet):
    """
    Check if the sample sequence can match the range shapelet at some position.

    Args:
        sample: Sample sequence (1D array)
        range_shapelet: Range shapelet, format: [[min1, max1], [min2, max2], ...]

    Returns:
        bool: Whether matching is successful
    """
    sample_length = len(sample)
    shapelet_length = len(range_shapelet)

    if shapelet_length > sample_length:
        return False

    # Traverse each possible starting position of the sample sequence
    for start_pos in range(sample_length - shapelet_length + 1):
        match = True

        # Check if each value of the sample at this position is within the corresponding range
        for i, (range_min, range_max) in enumerate(range_shapelet):
            sample_value = sample[start_pos + i]

            # Check if the sample value is within the range
            if not (range_min <= sample_value <= range_max):
                match = False
                break

        if match:
            return True

    return False


def calculate_range_shapelet_coverage(range_shapelet, data_samples):
    """
    Calculate the coverage rate of range shapelet in given data samples.

    Args:
        range_shapelet: Range shapelet, format: [[min1, max1], [min2, max2], ...]
        data_samples: List of data samples

    Returns:
        float: Coverage rate (matched samples / total samples)
    """
    if len(data_samples) == 0:
        return 0.0

    matched_count = 0
    for sample in data_samples:
        if check_range_shapelet_match(sample, range_shapelet):
            matched_count += 1

    return matched_count / len(data_samples)



def _filter_shapelets_with_dual_representation(shapelets, train_seq_datagen, class_name,
                                               sample_num_target, sample_num_other, score):
    """
    Range shapelet filter: keep shapelets where intra-class coverage is greater than all other class coverage.
    Support shapelet dictionaries with multiple representation forms.

    Args:
        shapelets: List of shapelets, each element is a dictionary:
                  {'range': [[min1, max1], [min2, max2], ...],
                   'mean_std': [[mean1, std1], [mean2, std2], ...]}
        train_seq_datagen: Training data generator
        class_name: Target class name
        sample_num_target: Number of samples for target class
        sample_num_other: Number of samples for other classes
        score: Score threshold

    Returns:
        filtered_results: List of filtered results, each element is a (shapelet_dict, coverage_score) tuple
    """
    print(f"Starting filtering, original shapelet count: {len(shapelets)}")

    # 1. Get all class names, excluding target class
    all_class_names = train_seq_datagen.class_names
    other_class_names = [cls for cls in all_class_names if cls != class_name]

    print(f"Target class: {class_name}")
    # print(f"Other classes: {other_class_names}")

    # 2. Load target class data
    target_data, *_ = train_seq_datagen.load_data_from_single_class(
        class_name, sample_num_target
    )
    print(f"Target class samples: {len(target_data)}")

    if len(other_class_names) == 0:
        print("⚠ No other classes, return all shapelets (score as maximum value)")
        return [(shapelet, 1000) for shapelet in shapelets]

    # 3. Load data for each other class
    other_classes_data = {}
    max_class_num = max(train_seq_datagen.class_nums.values())
    ratio = sample_num_other / max_class_num
    for other_class in other_class_names:
        sample_num = int(train_seq_datagen.class_nums[other_class] * ratio)
        other_data, *_ = train_seq_datagen.load_data_from_single_class(
            other_class, sample_num
        )
        other_classes_data[other_class] = other_data

    print(f"Other class samples loaded")

    # 4. Check each shapelet (using range form for coverage calculation)
    filtered_results = []
    rejected_count = 0
    all_shapelet_scores = []  # Store score information for all shapelets as backup

    for i, shapelet_dict in enumerate(shapelets):
        # Extract range representation for coverage calculation
        range_shapelet = shapelet_dict['range']

        # Calculate target class coverage
        target_class_coverage = calculate_range_shapelet_coverage(
            range_shapelet, target_data
        )

        # Calculate coverage for each other class
        other_coverages = {}
        for other_class, other_data in other_classes_data.items():
            other_coverage = calculate_range_shapelet_coverage(
                range_shapelet, other_data
            )
            other_coverages[other_class] = other_coverage
        #
        # # Find the maximum and average coverage among other classes
        max_other_coverage = max(other_coverages.values()) if other_coverages else 0.0
        weight = [i.shape[0] for i in other_classes_data.values()]
        mean_other_coverage = np.average(
            list(other_coverages.values()), weights=weight
        ) if other_coverages else 0.0

        # Store score information for all shapelets
        all_shapelet_scores.append(
            (i, max_other_coverage, shapelet_dict, target_class_coverage)
        )

        # Calculate score
        score_1 = int(target_class_coverage / (mean_other_coverage + 0.0001))
        score_2 = target_class_coverage / (max_other_coverage + 0.0001)

        if score_2 > score:
            # Keep the entire shapelet_dict (contains range and mean_std)
            filtered_results.append((shapelet_dict, int(score_1)))
        else:
            rejected_count += 1


    # 5. Output statistics
    reduction_ratio = rejected_count / len(shapelets) if len(shapelets) > 0 else 0

    print(f"\nFiltering completed:")
    print(f"  Original count: {len(shapelets)}")
    print(f"  Kept count: {len(filtered_results)}")
    print(f"  Rejected count: {rejected_count}")
    print(f"  Rejection ratio: {reduction_ratio:.1%}")

    if len(filtered_results) == 0:
        print("⚠ All shapelets were filtered out, return the bottom 10% with lowest maximum coverage among other classes as backup")
        # Sort by maximum coverage among other classes, return the best 10%
        all_shapelet_scores.sort(key=lambda x: x[1])  # Sort by max_other_coverage
        backup_count = max(1, len(shapelets) // 10)
        backup_results = []
        for item in all_shapelet_scores[:backup_count]:
            _, max_other_coverage, shapelet_dict, _ = item
            score_val = int(1 / (max_other_coverage + 0.001))
            backup_results.append((shapelet_dict, score_val))

        print(f"  Return {len(backup_results)} shapelets with lowest maximum coverage among other classes as backup")
        return backup_results

    return filtered_results