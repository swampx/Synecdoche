import numpy as np
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from data_generator import DataGenerator
import yaml


config =yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
dataset = config['dataset']['name']
train_dir = config['dataset']['txt_folder'] + dataset
valid_dir = train_dir + '_v'
test_dir = train_dir + '_t'

total_length = config['rules_generating']['max_length']
num_classes = config['dataset']['num_classes']
batch_size = config['cnn_training']['batch_size']


# Use existing DataGenerator
train_seq_datagen = DataGenerator(data_dir=train_dir, enhancement=False, num_classes=num_classes,batch_size=batch_size,
                                  total_length=total_length,plus=True)

valid_seq_datagen = DataGenerator(data_dir=valid_dir, enhancement=False, num_classes=num_classes,batch_size=batch_size,
                                  total_length=total_length, plus=True)

test_seq_datagen = DataGenerator(data_dir=test_dir, enhancement=False,num_classes=num_classes,batch_size=batch_size,
                                 total_length=total_length, plus=True)


def extract_data_from_generator(datagen):
    """Extract all data from DataGenerator"""
    X_list = []
    y_list = []

    # Iterate through all batches
    for i in range(len(datagen)):
        batch_x, batch_y = datagen[i]
        X_list.append(batch_x)
        y_list.append(batch_y)

    # Concatenate all batches
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Convert one-hot encoding to class labels
    y_labels = np.argmax(y, axis=1)

    # For decision tree, flatten sequence data
    X_flattened = X.reshape(X.shape[0], -1)

    return X_flattened, y_labels


# Extract data
print("Extracting data from data generator...")
X_train, y_train = extract_data_from_generator(train_seq_datagen)
X_valid, y_valid = extract_data_from_generator(valid_seq_datagen)
X_test, y_test = extract_data_from_generator(test_seq_datagen)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_valid.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {num_classes}")

# Decision tree configuration options - you can adjust these parameters as needed
TREE_CONFIGS = [
    {
        'name': 'single_decision_tree_depth10',
        'model': DecisionTreeClassifier(max_depth=10, random_state=42,class_weight='balanced'),
        'type': 'single'
    },
]

# Create directory to save models

model_save_dir = config['cnn_training']['model_saved_path'] + f"{dataset}/decision_trees/"
os.makedirs(model_save_dir, exist_ok=True)

best_model = None
best_f1 = 0
best_config_name = ""

print("\n" + "=" * 60)
print("Starting to train decision tree models")
print("=" * 60)

for config in TREE_CONFIGS:
    model_name = config['name']
    model = config['model']

    print(f"\nTraining: {model_name}")
    print("-" * 40)

    # Train model
    model.fit(X_train, y_train)

    # Validation set prediction
    valid_pred = model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, valid_pred)
    valid_f1 = f1_score(y_valid, valid_pred, average='macro')

    # Test set prediction
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    # Display results
    print(f"Validation accuracy: {valid_accuracy:.4f}")
    print(f"Validation F1 score: {valid_f1:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")

    # Save model
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pkl")
    joblib.dump(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Record best model
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        best_model = model
        best_config_name = model_name

print("\n" + "=" * 60)
print("Best model detailed results")
print("=" * 60)

print(f"Best model: {best_config_name}")
print(f"Best validation F1 score: {best_f1:.4f}")

# Best model's detailed test set results
best_test_pred = best_model.predict(X_test)
best_test_accuracy = accuracy_score(y_test, best_test_pred)
best_test_f1 = f1_score(y_test, best_test_pred, average='macro')

print(f"\nBest model test set results:")
print(f"Test accuracy: {best_test_accuracy:.4f}")
print(f"Test F1 score: {best_test_f1:.4f}")

print(f"\nDetailed classification report:")
print(classification_report(y_test, best_test_pred))

# Save an additional copy of the best model
best_model_path = os.path.join(model_save_dir, f"best_model.pkl")
joblib.dump(best_model, best_model_path)
print(f"\nBest model additionally saved to: {best_model_path}")

# Save model metadata
model_info = {
    'best_model_name': best_config_name,
    'best_validation_f1': best_f1,
    'test_accuracy': best_test_accuracy,
    'test_f1': best_test_f1,
    'num_classes': num_classes,
    'input_shape': X_train.shape[1],
    'dataset': dataset
}

info_save_path = os.path.join(model_save_dir, "model_info.pkl")
with open(info_save_path, 'wb') as f:
    pickle.dump(model_info, f)

print(f"Model information saved to: {info_save_path}")

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)
print(f"Total trained {len(TREE_CONFIGS)} models")
print(f"All models saved in: {model_save_dir}")
print(f"Best model: {best_config_name}")
print(f"Best model test accuracy: {best_test_accuracy:.4f}")
print(f"Best model test F1 score: {best_test_f1:.4f}")