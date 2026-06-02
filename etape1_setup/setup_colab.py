"""
ÉTAPE 1 : Configuration Google Colab
Configurer l'environnement et installer les dépendances
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """
    Installer toutes les dépendances nécessaires
    """
    print("\n" + "="*60)
    print("📦 Installation des dépendances...")
    print("="*60)
    
    packages = [
        "mediapipe==0.10.21",
        "opencv-python==4.8.1.78",
        "tensorflow==2.14.0",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "streamlit==1.27.0",
        "pillow==10.0.0",
    ]
    
    for package in packages:
        print(f"\n📥 Installation de {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("\n✅ Toutes les dépendances sont installées !")


def create_project_structure():
    """
    Créer la structure des dossiers du projet
    """
    print("\n" + "="*60)
    print("📁 Création de la structure du projet...")
    print("="*60)
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "etape1_setup",
        "etape2_preprocessing",
        "etape3_model",
        "etape4_webcam",
        "etape5_streamlit",
        "notebooks",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Créé : {directory}")
    
    print("\n✅ Structure du projet créée !")


def check_gpu_colab():
    """
    Vérifier si on est sur Google Colab et si le GPU est disponible
    """
    print("\n" + "="*60)
    print("🔍 Vérification de l'environnement...")
    print("="*60)
    
    # Vérifier Google Colab
    try:
        import google.colab
        print("✅ Exécution sur Google Colab")
        is_colab = True
    except ImportError:
        print("⚠️  Exécution locale (pas Colab)")
        is_colab = False
    
    # Vérifier GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU disponible : {len(gpus)} GPU(s) trouvé(s)")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("⚠️  Aucun GPU détecté")
    except Exception as e:
        print(f"❌ Erreur lors de la vérification GPU : {e}")
    
    # Vérifier versions
    print("\n📋 Versions des librairies :")
    
    try:
        import mediapipe
        print(f"   - MediaPipe : {mediapipe.__version__}")
    except:
        print("   - MediaPipe : Non installé")
    
    try:
        import cv2
        print(f"   - OpenCV : {cv2.__version__}")
    except:
        print("   - OpenCV : Non installé")
    
    try:
        import tensorflow as tf
        print(f"   - TensorFlow : {tf.__version__}")
    except:
        print("   - TensorFlow : Non installé")
    
    try:
        import numpy as np
        print(f"   - NumPy : {np.__version__}")
    except:
        print("   - NumPy : Non installé")


def main():
    """
    Fonction principale pour configurer tout
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "🎬 LIP READING PROJECT - ÉTAPE 1" + " "*12 + "#")
    print("#" + " "*20 + "Configuration & Setup" + " "*18 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    try:
        # 1. Créer la structure
        create_project_structure()
        
        # 2. Installer les dépendances
        install_dependencies()
        
        # 3. Vérifier l'environnement
        check_gpu_colab()
        
        print("\n" + "="*60)
        print("🎉 ÉTAPE 1 TERMINÉE AVEC SUCCÈS !")
        print("="*60)
        print("\n✨ Vous pouvez maintenant passer à l'ÉTAPE 2 :")
        print("   → Extraction des lèvres (Preprocessing)")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
