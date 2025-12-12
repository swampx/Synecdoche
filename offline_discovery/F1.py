import tensorflow as tf

class F1MicroScore(tf.keras.metrics.Metric):
    def __init__(self, thresholds=0.5, name='f1_score', **kwargs):
        super(F1MicroScore, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, self.thresholds), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred, axis=0)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        # Accumulate counts
        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


class F1MacroScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes, thresholds=0.5, name='f1_macro_score', **kwargs):
        super(F1MacroScore, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.thresholds = thresholds
        # 为每个类别分别计算TP、FP、FN
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, self.thresholds), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # 计算每个类别的TP、FP、FN
        true_positives = tf.reduce_sum(y_true * y_pred, axis=0)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        # 为每个类别分别累加
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        # 计算每个类别的precision和recall
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

        # 计算每个类别的F1分数
        f1_per_class = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

        # 返回所有类别F1分数的平均值（macro平均）
        return tf.reduce_mean(f1_per_class)

    def reset_states(self):
        self.true_positives.assign(tf.zeros((self.num_classes,)))
        self.false_positives.assign(tf.zeros((self.num_classes,)))
        self.false_negatives.assign(tf.zeros((self.num_classes,)))