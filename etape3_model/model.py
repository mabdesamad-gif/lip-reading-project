"""
ÉTAPE 3 : Architecture du Modèle Deep Learning
CNN + LSTM pour la reconnaissance des lèvres
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (
    Conv3D, Conv2D, LSTM, Dense, Dropout, 
    BatchNormalization, MaxPooling3D, MaxPooling2D,
    Flatten, TimeDistributed, Reshape, Input, concatenate
)
from tensorflow.keras.models import Model
import numpy as np


def create_cnn_lstm_model(input_shape=(20, 96, 48, 1), num_classes=500):
    """
    Créer un modèle CNN + LSTM pour le lip reading
    
    Architecture :
    - Conv3D : Extraction de features spatio-temporelles
    - LSTM : Modélisation des dépendances temporelles
    - Dense : Classification finale
    
    Args:
        input_shape: Shape de l'input (frames, height, width, channels)
        num_classes: Nombre de mots à prédire
    
    Returns:
        model: Modèle Keras compilé
    """
    
    model = Sequential([
        # Conv3D - Extraction features spatio-temporelles
        Conv3D(32, kernel_size=(3, 5, 5), activation='relu', padding='same',
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.2),
        
        # Conv3D - Extraction features plus complexes
        Conv3D(64, kernel_size=(3, 5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2)),
        Dropout(0.2),
        
        # Conv3D - Extraction features hautes dimensions
        Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Reshape pour LSTM (frames, features)
        Reshape((input_shape[0], -1)),
        
        # LSTM - Modélisation temporelle
        LSTM(256, return_sequences=True, dropout=0.2),
        LSTM(256, dropout=0.2),
        
        # Classification dense
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_conv2d_lstm_model(input_shape=(20, 96, 48, 1), num_classes=500):
    """
    Modèle alternatif : TimeDistributed Conv2D + LSTM
    Plus léger et souvent plus stable
    
    Args:
        input_shape: Shape de l'input (frames, height, width, channels)
        num_classes: Nombre de mots à prédire
    
    Returns:
        model: Modèle Keras compilé
    """
    
    model = Sequential([
        # TimeDistributed Conv2D - CNN par frame
        TimeDistributed(
            Conv2D(32, kernel_size=(3, 5), activation='relu', padding='same'),
            input_shape=input_shape
        ),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.2)),
        
        # TimeDistributed Conv2D - Plus de features
        TimeDistributed(
            Conv2D(64, kernel_size=(3, 5), activation='relu', padding='same')
        ),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.2)),
        
        # TimeDistributed Flatten
        TimeDistributed(Flatten()),
        
        # LSTM - Modélisation temporelle
        LSTM(256, return_sequences=True, dropout=0.2),
        LSTM(256, dropout=0.2),
        
        # Classification
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_simple_model(input_shape=(20, 96, 48, 1), num_classes=500):
    """
    Modèle simple et rapide pour le debugging
    
    Args:
        input_shape: Shape de l'input
        num_classes: Nombre de classes
    
    Returns:
        model: Modèle simplifié
    """
    
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', 
               input_shape=(96, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compiler le modèle avec optimiseur et loss function
    
    Args:
        model: Modèle Keras
        learning_rate: Taux d'apprentissage
    
    Returns:
        model: Modèle compilé
    """
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


def get_model_summary(model):
    """
    Afficher un résumé du modèle
    
    Args:
        model: Modèle Keras
    """
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DU MODÈLE")
    print("="*80)
    model.summary()
    print(f"\n📈 Nombre total de paramètres : {model.count_params():,}")
    print("="*80 + "\n")


def main():
    """
    Démonstration de création de modèles
    """
    print("\n" + "="*60)
    print("🤖 ÉTAPE 3 : Architecture du Modèle Deep Learning")
    print("="*60)
    
    # Paramètres
    input_shape = (20, 96, 48, 1)  # (frames, height, width, channels)
    num_classes = 500  # LRW dataset
    
    print(f"\n📐 Configuration :")
    print(f"   Input shape : {input_shape}")
    print(f"   Nombre de classes : {num_classes}")
    
    # Créer le modèle CNN+LSTM
    print("\n🏗️  Création du modèle CNN+LSTM...")
    model = create_cnn_lstm_model(input_shape, num_classes)
    
    # Compiler le modèle
    print("⚙️  Compilation du modèle...")
    model = compile_model(model, learning_rate=0.001)
    
    # Afficher le résumé
    get_model_summary(model)
    
    print("✅ Modèle prêt pour l'entraînement !")
    
    # Tester avec des données dummy
    print("\n🧪 Test avec des données dummy...")
    dummy_input = np.random.randn(2, 20, 96, 48, 1).astype(np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"✅ Output shape : {dummy_output.shape}")
    print(f"   Prédictions : {dummy_output}")


if __name__ == "__main__":
    main()
