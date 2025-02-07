import os
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large, ResNet50V2, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
from datetime import datetime

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_input(x):
    """Preprocess input images by scaling them to [0,1]."""
    return x / 255.0

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='glorot_uniform',
                                trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        # Add epsilon for numerical stability
        attention_weights = tf.nn.softmax(tf.matmul(x, self.W) + tf.keras.backend.epsilon(), axis=1)
        attended_output = x * attention_weights
        return attended_output

class CoffeeLeafClassifier:
    def __init__(self, data_path, img_size=224, batch_size=32):
        
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataset_splits = self._detect_splits()
        self.class_names = self._detect_classes()
        self.num_classes = len(self.class_names)
        self._setup_directories()
        self.model = None

    def _detect_splits(self):
        splits = []
        potential_splits = ['train', 'valid', 'validation', 'test']
        for split in potential_splits:
            if os.path.exists(os.path.join(self.data_path, split)):
                splits.append(split)
        if not splits:
            raise ValueError(f"No valid dataset splits found in {self.data_path}")
        return splits

    def _detect_classes(self):
        train_path = os.path.join(self.data_path, 'train')
        if not os.path.exists(train_path):
            raise ValueError(f"Training directory not found at {train_path}")
            
        classes = set()
        for split in self.dataset_splits:
            split_path = os.path.join(self.data_path, split)
            classes.update([d for d in os.listdir(split_path)
                          if os.path.isdir(os.path.join(split_path, d))])
        if not classes:
            raise ValueError("No class directories found")
        return sorted(list(classes))

    def _setup_directories(self):
        base_dir = os.path.dirname(self.data_path)
        self.output_dir = os.path.join(base_dir, 'output')
        self.saved_models_dir = os.path.join(base_dir, 'saved_models')
        
        for directory in [self.output_dir, self.saved_models_dir]:
            os.makedirs(directory, exist_ok=True)

    def create_data_generators(self):
        """Create data generators for training, validation, and test sets without augmentation"""
        # Use same preprocessing for all splits since data is already augmented
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        # Create generators for each split
        self.train_generator = datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.validation_generator = datagen.flow_from_directory(
            os.path.join(self.data_path, 'validation'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.test_generator = datagen.flow_from_directory(
            os.path.join(self.data_path, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Calculate steps
        self.train_steps = (self.train_generator.samples + self.batch_size - 1) // self.batch_size
        self.validation_steps = (self.validation_generator.samples + self.batch_size - 1) // self.batch_size
        self.test_steps = (self.test_generator.samples + self.batch_size - 1) // self.batch_size

    def build_model(self):
        input_shape = (self.img_size, self.img_size, 3)
        input_layer = layers.Input(shape=input_shape)

        # MobileNetV3Large branch
        mobilenet = MobileNetV3Large(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        mobilenet.trainable = False
        
        # ResNet50V2 branch
        resnet = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        resnet.trainable = False
        
        # Process through MobileNet
        x1 = mobilenet(input_layer)
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(512, activation='relu')(x1)
        x1 = layers.Dropout(0.5)(x1)
        
        # Process through ResNet
        x2 = resnet(input_layer)
        x2 = layers.GlobalAveragePooling2D()(x2)
        x2 = layers.Dense(512, activation='relu')(x2)
        x2 = layers.Dropout(0.5)(x2)
        
        # Concatenate both models - This is where ensemble happens
        merged = layers.Concatenate()([x1, x2])
        
        # Add attention layer to weight features
        attention_output = AttentionLayer()(merged)
        
        # Final dense layers
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(attention_output)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        return model

    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))

    def evaluate_model(self):
        """Evaluate the model and save all metrics."""
        val_predictions = []
        val_true_labels = []
        
        for i in range(self.validation_steps):
            x, y = next(self.validation_generator)
            pred = self.model.predict(x)
            val_predictions.extend(np.argmax(pred, axis=1))
            val_true_labels.extend(np.argmax(y, axis=1))
        
        return val_true_labels, val_predictions

    def save_evaluation_metrics(self, y_true, y_pred, history):
        """Save all evaluation metrics and visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_dir = os.path.join(self.output_dir, f'evaluation_{timestamp}')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save classification report
        report = classification_report(y_true, y_pred, 
                                    target_names=self.class_names,
                                    output_dict=True)
        with open(os.path.join(metrics_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save ROC curves
        plt.figure(figsize=(10, 8))
        y_true_bin = tf.keras.utils.to_categorical(y_true)
        y_pred_prob = tf.keras.utils.to_categorical(y_pred)
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
        y_pred_prob = self.model.predict(self.validation_generator, steps=self.validation_steps)
        
        plt.plot([0, 1], [0, 1], 'k--')
        y_pred_prob = self.model.predict(self.validation_generator, steps=self.validation_steps)
        y_pred_prob = tf.keras.utils.to_categorical(np.argmax(y_pred_prob, axis=1))
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {self.class_names[i]} (area = {roc_auc:.2f})')
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(metrics_dir, 'training_history.csv'))
        
        # Save training plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_curves.png'))
        plt.close()
        
        # Save model summary
        with open(os.path.join(metrics_dir, 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return metrics_dir

    def train(self, epochs=20):  
        try:
            self.create_data_generators()
            self.model = self.build_model()
            
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.2, 
                    patience=5,  # Increased patience
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss', 
                    patience=12,  # Increased patience for early stopping
                    restore_best_weights=True, 
                    min_delta=1e-4,
                    verbose=1
                ),
                ModelCheckpoint(
                    os.path.join(self.saved_models_dir, 'best_ensemble_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
            ]

            history = self.model.fit(
                self.train_generator,
                validation_data=self.validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                workers=1,
                use_multiprocessing=False,
                verbose=1
            )

            self.model.save(os.path.join(self.saved_models_dir, 'final_model.h5'))
            return history

        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.model is not None:
                del self.model

def main():
    try:
        data_path = r"C:\Users\athar\OneDrive\Desktop\aug\dataset"
        if not os.path.exists(data_path):
            raise ValueError(f"Dataset path not found: {data_path}")
            
        classifier = CoffeeLeafClassifier(data_path)
        history = classifier.train()
        y_true, y_pred = classifier.evaluate_model()
        
        metrics_dir = classifier.save_evaluation_metrics(y_true, y_pred, history)
        print(f"\nEvaluation metrics and visualizations saved to: {metrics_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()