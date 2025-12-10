from fruit import FruitRecognitionCNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

def create_dataframe_from_filenames(directory):
    """
    Buat dataframe dari nama file
    Format nama file: apple_1.jpg, orange_2.jpg, etc
    """
    filenames = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract label dari nama file (sebelum underscore)
            label = filename.split('_')[0]
            filenames.append(os.path.join(directory, filename))
            labels.append(label)
    
    df = pd.DataFrame({
        'filename': filenames,
        'class': labels
    })
    
    return df

if __name__ == "__main__":
    # Inisialisasi model
    fruit_model = FruitRecognitionCNN(img_size=(224, 224))

    # Create dataframes dari filenames
    train_df = create_dataframe_from_filenames('dataset/fruit/train')
    test_df = create_dataframe_from_filenames('dataset/fruit/test')
    
    # Data augmentation untuk training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Test data (no augmentation)
    test_datagen = ImageDataGenerator()
    
    # Create generators
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'  # 80% untuk training
    )
    
    val_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # 20% untuk validasi
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Update num_classes di model
    fruit_model.num_classes = len(train_df['class'].unique())
    fruit_model.class_names = sorted(train_df['class'].unique())
    
    # Create and compile model
    fruit_model.create_model()
    fruit_model.compile_model(learning_rate=0.001)
    
    # Training
    print("\n[1/2] Training model...")
    fruit_model.train(train_gen, val_gen, epochs=20)
    
    # Fine-tuning
    print("\n[2/2] Fine-tuning model...")
    fruit_model.fine_tune(train_gen, val_gen, epochs=10)
    
    # Plot training history
    fruit_model.plot_training_history()
    
    # Evaluate
    print("\n[3/3] Evaluating on test set...")
    fruit_model.evaluate(test_gen)
    
    # Save model
    fruit_model.save_model('train_results/fruit_recognition_model.h5')
    
    print("\nTraining complete! Model saved.")