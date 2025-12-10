import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class FruitRecognitionCNN:
    def __init__(self, img_size=(224, 224), num_classes=None):
        """
        Inisialisasi model pengenalan buah
        
        Args:
            img_size: ukuran input gambar (height, width)
            num_classes: jumlah kelas buah yang akan dikenali
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None
        
    def create_model(self):
        """
        Membuat model menggunakan pre-trained MobileNetV2
        dengan transfer learning approach
        """
        # Load pre-trained MobileNetV2 tanpa top layer
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers untuk transfer learning
        base_model.trainable = False
        
        # Tambahkan custom layers untuk klasifikasi buah
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing untuk MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create final model
        self.model = keras.Model(inputs, outputs)
        
        print("Model berhasil dibuat!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model dengan optimizer dan loss function
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model berhasil di-compile!")
    
    def prepare_data(self, train_dir, val_dir, batch_size=32):
        """
        Mempersiapkan data training dan validation menggunakan ImageDataGenerator
        
        Args:
            train_dir: direktori data training
            val_dir: direktori data validation
            batch_size: ukuran batch
        
        Returns:
            train_generator, val_generator
        """
        # Data augmentation untuk training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data hanya rescaling
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Simpan class names
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Kelas yang ditemukan: {self.class_names}")
        print(f"Jumlah kelas: {self.num_classes}")
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=20):
        """
        Training model
        
        Args:
            train_generator: generator untuk data training
            val_generator: generator untuk data validation
            epochs: jumlah epoch training
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Training
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        print("Training selesai!")
        return self.history
        
    def fine_tune(self, train_generator, val_generator, epochs=10, unfreeze_layers=50):
        """
        Fine-tuning model dengan membuka beberapa layer base model
        """
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) and 'mobilenet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            print("Warning: Base model not found, skipping fine-tuning")
            return None
        
        # Unfreeze base model layers
        base_model.trainable = True
        
        # Freeze layers kecuali unfreeze_layers terakhir
        total_layers = len(base_model.layers)
        for i, layer in enumerate(base_model.layers):
            if i < total_layers - unfreeze_layers:
                layer.trainable = False
            else:
                layer.trainable = True
        
        print(f"Fine-tuning: Unfreezing last {unfreeze_layers} layers of {total_layers} total layers")
        
        # Recompile dengan learning rate lebih kecil
        self.compile_model(learning_rate=1e-5)
        
        history_fine = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        print("Fine-tuning selesai!")
        return history_fine
    
    def predict_image(self, img_path):
        """
        Prediksi kelas buah dari gambar
        
        Args:
            img_path: path ke file gambar
        
        Returns:
            predicted_class, confidence, all_probabilities
        """
        # Load dan preprocess gambar
        img = keras.preprocessing.image.load_img(
            img_path, 
            target_size=self.img_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def plot_training_history(self):
        """
        Visualisasi training history
        """
        if self.history is None:
            print("Belum ada history training!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('train_results/training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved!")
        return fig
    
    def evaluate(self, test_generator):
        """
        Evaluasi model pada test dataset
        
        Args:
            test_generator: generator untuk data testing
        """
        # Get predictions
        y_pred = self.model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                   target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('train_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved!")
        
        return y_pred_classes, y_true
    
    def save_model(self, filepath):
        """
        Simpan model ke file
        """
        self.model.save(filepath)
        print(f"Model disimpan ke {filepath}")
    
    def load_model(self, filepath):
        """
        Load model dari file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded dari {filepath}")