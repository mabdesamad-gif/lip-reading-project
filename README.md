# 👄 Lip Reading Project - Reconnaissance Visuelle de la Parole

Un projet complet de Deep Learning pour prédire ce qui est dit en analysant les mouvements des lèvres (sans son).

## 📋 Vue d'ensemble

Ce projet utilise :
- **MediaPipe** : Détection des 468 landmarks du visage
- **TensorFlow** : Architecture CNN + LSTM
- **OpenCV** : Traitement vidéo en temps réel
- **Streamlit** : Interface web interactive

## 🎯 Les 5 Étapes

### Étape 1 ✅ - Environnement Google Colab
- Configuration de l'environnement
- Installation des dépendances
- Structure des dossiers

### Étape 2 - Extraction des Lèvres
- Détection avec MediaPipe (468 landmarks)
- Isolation des 40 points des lèvres
- Crop 96×48px et séquences de 20 frames
- Normalisation et sauvegarde

### Étape 3 - Modèle Deep Learning
- Architecture CNN (extraction features par frame)
- LSTM (analyse temporelle)
- Entraînement sur GRID corpus
- Métriques : Accuracy, WER

### Étape 4 - Webcam Live
- Prédiction en temps réel
- Affichage d'emojis dynamiques
- Graphe de confiance en direct
- Overlay sur la vidéo

### Étape 5 - Interface Streamlit
- Dashboard complet
- Upload vidéo ou webcam live
- Historique des prédictions
- Visualisation émotions

## 📁 Structure du Projet

```
lip-reading-project/
├── README.md
├── requirements.txt
├── .gitignore
├── etape1_setup/
│   └── setup_colab.py
├── etape2_preprocessing/
│   ├── extract_lips.py
│   └── preprocess.py
├── etape3_model/
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── etape4_webcam/
│   ├── webcam_live.py
│   └── utils.py
├── etape5_streamlit/
│   ├── app.py
│   └── streamlit_utils.py
├── notebooks/
│   ├── Etape1_Setup.ipynb
│   ├── Etape2_Preprocessing.ipynb
│   ├── Etape3_Model_Training.ipynb
│   ├── Etape4_Webcam_Live.ipynb
│   └── Etape5_Streamlit.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── tests/
    └── test_preprocessing.py
```

## 🚀 Installation Rapide

```bash
# Cloner le repo
git clone https://github.com/mabdesamad-gif/lip-reading-project.git
cd lip-reading-project

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Guide d'Utilisation

### Sur Google Colab (Recommandé)
1. Ouvrir les notebooks dans `notebooks/`
2. Suivre les étapes une par une
3. Entraîner le modèle sur GPU Colab (gratuit)

### Sur Streamlit (Final)
```bash
cd etape5_streamlit
streamlit run app.py
```

## 📊 Dataset

- **GRID Corpus** (débutant) : 34 locuteurs, 1000 phrases, fond blanc
- **LRW** (avancé) : 500 mots, vidéos BBC

## 🎨 Fonctionnalités "WOW"

✨ Webcam live en temps réel
✨ Emoji dynamique par mot prédit
✨ Graphe de confiance live
✨ Interface Streamlit complète

## 📚 Ressources

- [MediaPipe Docs](https://mediapipe.dev/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [GRID Corpus](http://www.talkvision.com/GRID.html)
- [LRW Dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

## 👤 Auteur

**mabdesamad-gif** - Lip Reading Project

## 📝 Licence

MIT License - Libre d'utilisation

---

**Prêt à commencer ?** Allez à l'Étape 1 ! 🚀
