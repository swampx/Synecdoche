import os
import tensorflow as tf
from cnn_model import cnn_with_embedding
from F1 import F1MacroScore
from data_generator import DataGenerator
from tensorflow.keras.optimizers import Adam
from rule_generate import generate_rules
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# prepare data
config =yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
dataset = config['dataset']['name']
train_dir = config['dataset']['txt_folder'] + dataset
valid_dir = train_dir + '_v'
test_dir = train_dir + '_t'

total_length = config['preprocess']['max_pkt_number']
num_classes = config['dataset']['num_classes']
batch_size = config['cnn_training']['batch_size']

train_seq_datagen = DataGenerator(data_dir=train_dir, enhancement=False, batch_size=batch_size, num_classes=num_classes,
                                 total_length=total_length, plus=True)
valid_seq_datagen = DataGenerator(data_dir=valid_dir, enhancement=False, batch_size=batch_size, num_classes=num_classes,
                                  total_length=total_length, plus=True)
test_seq_datagen = DataGenerator(data_dir=test_dir, enhancement=False,  batch_size=batch_size, num_classes=num_classes,
                                 total_length=total_length,plus=True)

# build cnn_model
convs = config['cnn_training']['convs']
embed_dimension = config['cnn_training']['embed_dimension']
filter = config['cnn_training']['filter']
model_cnn = cnn_with_embedding(class_num=num_classes, input_shape=total_length, convs=2, embed_dimension=128,
                               filter=128)


model_cnn.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['CategoricalAccuracy', F1MacroScore(num_classes)])

checkpoint_save_path = config['cnn_training']['model_saved_path'] + dataset + "/" + dataset + ".ckpt"
cp_callback_save = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, verbose=1,
                                                      monitor='val_categorical_accuracy',
                                                      save_weights_only=True,
                                                      save_best_only=True
                                                      )
training_callbacks = [
    cp_callback_save,
    tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode='max'
    ),
]


# training_cnn_model
print('training cnn model')
epochs = config['cnn_training']['epochs']
model_cnn.fit(train_seq_datagen, callbacks=training_callbacks,epochs=epochs, validation_data=valid_seq_datagen)
# model_cnn.load_weights(checkpoint_save_path)
model_cnn.evaluate(test_seq_datagen)




# generating key segments
max_length = config['rules_generating']['max_length']
rules_path = config['rules_generating']['rules_path'] + dataset + '/'
max_sample_num = config['rules_generating']['max_sample_num']
eps = config['rules_generating']['eps']
ratio = config['rules_generating']['ratio']
score = config['rules_generating']['score']

if not os.path.exists(rules_path):
    os.mkdir(rules_path)


for class_name in os.listdir(train_dir):
    print(class_name)
    rules = generate_rules(model_cnn, train_seq_datagen, class_name, max_sample_num, rules_path, max_length=max_length, eps=eps,
                           ratio=ratio, score=score, representation_type='range')