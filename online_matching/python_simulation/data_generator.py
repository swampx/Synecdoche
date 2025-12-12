from tensorflow.keras.utils import Sequence, to_categorical
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import random


# loaded_cdf_interp = joblib.load('cdf_interp_botnet.pkl')


def to_1d(path, enhancement=True, inter_time=True, total_length=30, actual_length=30, plus=True):
    with open(path, 'r') as file:
        lines = file.readlines()
        data = []

        for line in lines:
            if line.strip().find(',') != -1:
                data.append(line.strip().split(','))
            else:
                data.append(line.strip().split(' '))
        data = [[value for value in line if value != ''] for line in data]
        data = np.array(data, dtype=np.float32)
        res_data = []

        i, j, p, n = 0, 0, 0, 0

        while i < total_length and j < len(data[0]):
            res_data.append(data[1][j] * data[2][j])
            i += 1
            j += 1
        res = []
        data_len = len(res_data)
        if data_len > actual_length:
            res_data = res_data[:actual_length]
        actual_length = min(actual_length, data_len)

        if enhancement:
            append_first = random.randint(0, total_length - actual_length)
        else:
            append_first = 0
        append_last = total_length - append_first - actual_length
        res.extend(0 for i in range(append_first))
        res.extend(res_data)
        res.extend(0 for i in range(append_last))
        res = np.asarray(res)
        if plus:
            res += 1514

    return res


class DataGenerator(Sequence):
    def __init__(self, data_dir, num_classes, batch_size=32, shuffle=True, enhancement=False, inter_time=False,
                 total_length=30, actual_length=30, plus=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_names = sorted(os.listdir(data_dir))
        self.class_paths = [os.path.join(data_dir, class_name) for class_name in self.class_names]
        self.file_paths = []
        self.class_nums={}
        self.shuffle = shuffle
        self.enhancement = enhancement
        self.inter_time = inter_time
        self.total_length = total_length
        self.actual_length = actual_length

        self.plus = plus

        for class_path in self.class_paths:
            class_files = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.endswith('.txt')]
            self.file_paths.extend(class_files)
            self.class_nums[os.path.basename(class_path)]=len(class_files)

        self.num_samples = len(self.file_paths)
        self.indexes = np.arange(self.num_samples)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []

        for file_path in batch_files:
            res_data = to_1d(file_path, enhancement=self.enhancement, total_length=self.total_length,
                             actual_length=self.actual_length,
                             inter_time=self.inter_time, plus=self.plus,
                            )
            # res_data = to_1d_seq(file_path, length=self.length)
            batch_data.append(res_data)
            class_name = os.path.basename(os.path.dirname(file_path))
            class_label = self.class_names.index(class_name)
            batch_labels.append(class_label)

        batch_data = np.array(batch_data)
        batch_labels = to_categorical(batch_labels, num_classes=self.num_classes)
        batch_data = np.expand_dims(batch_data, axis=-1)  # Expand dimensions to make it 3D
        return batch_data, batch_labels

    def get_item_with_flow_length(self, idx):
        batch_files = self.file_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        batch_flow_lengths = []

        for file_path in batch_files:
            res_data = to_1d(
                file_path,
                enhancement=self.enhancement,
                total_length=self.total_length,
                actual_length=self.actual_length,
                inter_time=self.inter_time,
                plus=self.plus
            )
            basename = os.path.splitext(os.path.basename(file_path))[0]
            flow_length = int(basename.split('_')[1])

            batch_data.append(res_data)
            class_name = os.path.basename(os.path.dirname(file_path))
            class_label = self.class_names.index(class_name)
            batch_labels.append(class_label)
            batch_flow_lengths.append(flow_length)

        batch_data = np.array(batch_data)
        batch_labels = to_categorical(batch_labels, num_classes=self.num_classes)
        batch_data = np.expand_dims(batch_data, axis=-1)
        batch_flow_lengths = np.array(batch_flow_lengths)

        return batch_data, batch_labels, batch_flow_lengths

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_paths)

    def load_data_from_single_class(self, class_name, num_samples):
        class_dir = os.path.join(self.data_dir, class_name)
        all_files = os.listdir(class_dir)
        if num_samples == -1:
            selected_files = all_files
        else:
            num_samples = min(num_samples, len(all_files))
            selected_files = np.random.choice(all_files, num_samples, replace=False)
        datalist = []
        name = []
        for file_path in selected_files:
            res_data = to_1d(os.path.join(class_dir, file_path), enhancement=self.enhancement,
                             total_length=self.total_length,
                             actual_length=self.actual_length,
                             inter_time=self.inter_time, plus=self.plus)
            datalist.append(res_data)
            name.append(class_name)
        res = np.asarray(datalist)
        res = np.expand_dims(res, axis=-1)
        return res, name

    def load_data_from_other_classes(self, excluded_class, num_samples):
        # 获取所有类别
        all_classes = os.listdir(self.data_dir)

        # 移除排除的类别
        if excluded_class in all_classes:
            all_classes.remove(excluded_class)

        # 创建一个列表，用于存储所有剩余类别的样本文件路径
        all_other_files = []
        for selected_class in all_classes:
            class_dir = os.path.join(self.data_dir, selected_class)
            all_files = os.listdir(class_dir)
            all_other_files.extend([os.path.join(class_dir, file_path) for file_path in all_files])

        # 确保要获取的样本数不超过可用的样本数
        num_samples = min(num_samples, len(all_other_files))

        # 随机选择样本文件路径
        selected_files = random.sample(all_other_files, num_samples)

        datalist = []
        name = []

        for file_path in selected_files:
            res_data = to_1d(file_path, enhancement=self.enhancement, total_length=self.total_length,
                             actual_length=self.actual_length,
                             inter_time=self.inter_time, plus=self.plus)
            datalist.append(res_data)
            # name.append(os.path.basename(file_path))
            # 返回类别名称
            name.append(file_path.split('/')[1])
        res = np.asarray(datalist)
        res = np.expand_dims(res, axis=-1)

        return res, name

    def load_all_data(self, num_samples):
        all_classes = os.listdir(self.data_dir)
        # 创建一个列表，用于存储所有剩余类别的样本文件路径
        all_other_files = []
        for selected_class in all_classes:
            class_dir = os.path.join(self.data_dir, selected_class)
            all_files = os.listdir(class_dir)
            all_other_files.extend([os.path.join(class_dir, file_path) for file_path in all_files])
        if num_samples == -1:
            num_samples = len(all_other_files)
        if num_samples > len(all_other_files):
            num_samples = len(all_other_files)
        datalist = []
        name = []
        selected_files = random.sample(all_other_files, num_samples)
        for file_path in selected_files:
            file_path = file_path.replace('\\', '/')
            res_data = to_1d(file_path, enhancement=self.enhancement, total_length=self.total_length,
                             actual_length=self.actual_length,
                             inter_time=self.inter_time, plus=self.plus)
            # res_data = to_1d_seq(file_path, length=self.length)
            datalist.append(res_data)
            # name.append(os.path.basename(file_path))
            # 返回类别名称
            name.append(file_path.split('/')[-2])
        res = np.asarray(datalist)
        res = np.expand_dims(res, axis=-1)

        return res, name

    def get_all_integer_labels(self):
        all_labels = []
        for file_path in self.file_paths:
            class_name = os.path.basename(os.path.dirname(file_path))
            class_label = self.class_names.index(class_name)
            all_labels.append(class_label)
        return all_labels

    def create_binary_classification_dataset(self, positive_class_name, n, m):
        # Load data for the positive class
        positive_data, positive_names = self.load_data_from_single_class(positive_class_name, n)
        positive_labels = np.ones(len(positive_data))

        # Load data for the negative class
        negative_data, negative_names = self.load_data_from_other_classes(positive_class_name, m)
        negative_labels = np.zeros(len(negative_data))

        # Combine data and labels
        combined_data = np.concatenate((positive_data, negative_data), axis=0)
        combined_labels = np.concatenate((positive_labels, negative_labels), axis=0)
        combined_labels = to_categorical(combined_labels, num_classes=2)
        # Shuffle the dataset
        indices = np.arange(len(combined_labels))
        np.random.shuffle(indices)
        combined_data = combined_data[indices]
        combined_labels = combined_labels[indices]

        # Here you can split the dataset into training and validation sets if needed
        # train_data, val_data, train_labels, val_labels = train_test_split(combined_data, combined_labels, test_size=0.2)

        # Preprocess data if needed
        # processed_data = preprocess(combined_data)

        # Return the dataset ready for TensorFlow training
        return combined_data, combined_labels