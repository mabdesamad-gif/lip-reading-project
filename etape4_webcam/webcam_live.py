"""
ÉTAPE 4 : Interface Webcam Live
Capture en temps réel, prédiction, emoji dynamique et graphe de confiance
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
import threading


class WebcamLipReader:
    """
    Classe pour faire de la lecture de lèvres en temps réel via webcam
    """
    
    def __init__(self, model, word_list=None, sequence_length=20, confidence_threshold=0.5):
        """
        Initialiser le lecteur de lèvres webcam
        
        Args:
            model: Modèle TensorFlow entraîné
            word_list: Liste des mots (vocabulaire)
            sequence_length: Longueur des séquences (défaut 20)
            confidence_threshold: Seuil de confiance minimum
        """
        self.model = model
        self.word_list = word_list or [f"word_{i}" for i in range(model.output_shape[1])]
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialiser MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Indices des lèvres
        self.lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            178, 200, 199, 175, 171, 77, 90, 180, 85, 16,
        ]
        
        # Historique des prédictions
        self.prediction_history = deque(maxlen=5)
        self.confidence_history = deque(maxlen=30)
        
        # Émojis par mot (à personnaliser)
        self.emoji_dict = {
            'hello': '👋',
            'yes': '✅',
            'no': '❌',
            'thank': '🙏',
            'please': '🤲',
            'sorry': '😔',
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'love': '❤️'
        }
    
    def get_emoji_for_word(self, word):
        """
        Obtenir l'emoji correspondant à un mot
        
        Args:
            word: Mot prédit
        
        Returns:
            emoji: Emoji correspondant ou '🤔' par défaut
        """
        word_lower = word.lower()
        for key, emoji in self.emoji_dict.items():
            if key in word_lower:
                return emoji
        return '🤔'
    
    def extract_lips_frame(self, frame):
        """
        Extraire les lèvres d'une seule frame
        
        Args:
            frame: Image (numpy array)
        
        Returns:
            lip_crop: Région des lèvres crop (96x48x1)
            success: Booléen
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None, False
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Obtenir les points des lèvres
            lip_points = np.array([
                [landmarks[idx].x * w, landmarks[idx].y * h]
                for idx in self.lip_indices
            ], dtype=np.int32)
            
            # Bounding box
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop et resize
            lip_region = frame[y_min:y_max, x_min:x_max]
            if lip_region.size == 0:
                return None, False
            
            lip_crop = cv2.resize(lip_region, (96, 48))
            lip_crop = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
            lip_crop = np.expand_dims(lip_crop, axis=-1)
            lip_crop = lip_crop.astype(np.float32) / 255.0
            
            return lip_crop, True
            
        except Exception as e:
            print(f"Erreur extraction lèvres : {e}")
            return None, False
    
    def predict_sequence(self, sequence):
        """
        Prédire le mot d'une séquence
        
        Args:
            sequence: Séquence de frames (20, 96, 48, 1)
        
        Returns:
            word: Mot prédit
            confidence: Confiance de la prédiction
        """
        if sequence.shape[0] != self.sequence_length:
            return None, 0.0
        
        # Ajouter batch dimension
        input_data = np.expand_dims(sequence, axis=0)
        
        # Prédire
        predictions = self.model.predict(input_data, verbose=0)[0]
        confidence = np.max(predictions)
        word_idx = np.argmax(predictions)
        
        word = self.word_list[word_idx] if word_idx < len(self.word_list) else f"word_{word_idx}"
        
        return word, float(confidence)
    
    def draw_info_on_frame(self, frame, word, confidence, emoji):
        """
        Ajouter les informations sur la frame
        
        Args:
            frame: Image (numpy array)
            word: Mot prédit
            confidence: Confiance
            emoji: Emoji à afficher
        
        Returns:
            frame: Frame modifiée
        """
        h, w = frame.shape[:2]
        
        # Fond semi-transparent pour le texte
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Texte
        cv2.putText(frame, f"Word: {word}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Emoji
        cv2.putText(frame, emoji, (w-100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return frame
    
    def draw_confidence_bars(self, predictions, top_n=5):
        """
        Créer une image avec les barres de confiance
        
        Args:
            predictions: Vecteur de prédictions
            top_n: Nombre de top prédictions à afficher
        
        Returns:
            img: Image matplotlib en numpy array
        """
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        top_words = [self.word_list[i] for i in top_indices]
        top_scores = predictions[top_indices]
        
        fig = Figure(figsize=(8, 4))
        ax = fig.subplots()
        
        colors = ['green' if i == 0 else 'blue' for i in range(len(top_words))]
        ax.barh(top_words, top_scores, color=colors, alpha=0.7)
        ax.set_xlabel('Confiance')
        ax.set_title('Top 5 Prédictions')
        ax.set_xlim([0, 1])
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def run_webcam(self, show_confidence_bars=True):
        """
        Lancer la capture webcam en temps réel
        
        Args:
            show_confidence_bars: Afficher les barres de confiance
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'accéder à la webcam")
            return
        
        print("\n" + "="*60)
        print("🎥 WEBCAM LIVE - Lip Reading en temps réel")
        print("="*60)
        print("\n🎯 Commandes :")
        print("   'q' : Quitter")
        print("   'r' : Réinitialiser")
        print("\n")
        
        lip_sequence = deque(maxlen=self.sequence_length)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Redimensionner pour la performance
            frame = cv2.resize(frame, (960, 720))
            
            # Extraire les lèvres
            lip_crop, success = self.extract_lips_frame(frame)
            
            if success and lip_crop is not None:
                lip_sequence.append(lip_crop)
                
                # Prédire quand on a une séquence complète
                if len(lip_sequence) == self.sequence_length:
                    sequence = np.array(lip_sequence)
                    word, confidence = self.predict_sequence(sequence)
                    
                    if confidence > self.confidence_threshold:
                        emoji = self.get_emoji_for_word(word)
                        self.prediction_history.append(word)
                        self.confidence_history.append(confidence)
                        
                        # Afficher les infos
                        frame = self.draw_info_on_frame(frame, word, confidence, emoji)
            
            # Afficher l'historique
            if self.prediction_history:
                y_offset = 120
                cv2.putText(frame, "Historique:", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, word in enumerate(self.prediction_history):
                    cv2.putText(frame, f"• {word}", (20, y_offset + 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Afficher le nombre de frames
            cv2.putText(frame, f"Buffer: {len(lip_sequence)}/{self.sequence_length}",
                       (960-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            
            # Afficher la frame
            cv2.imshow('Lip Reading - Webcam Live', frame)
            
            # Afficher les barres de confiance
            if show_confidence_bars and self.confidence_history:
                # Créer un graphe simple
                graph = np.zeros((200, 640, 3), dtype=np.uint8)
                max_history = len(self.confidence_history)
                
                for i, conf in enumerate(self.confidence_history):
                    x = int((i / max_history) * 640)
                    y = int(200 - (conf * 200))
                    cv2.circle(graph, (x, y), 3, (0, 255, 0), -1)
                
                cv2.imshow('Confidence History', graph)
            
            # Commandes clavier
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                lip_sequence.clear()
                self.prediction_history.clear()
                print("🔄 Réinitialisé")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Capture webcam terminée")


def main():
    """
    Démonstration
    """
    print("\n" + "="*60)
    print("📺 ÉTAPE 4 : Interface Webcam Live")
    print("="*60)
    
    print("\n⚠️  Cette démonstration nécessite un modèle entraîné.")
    print("Utilisez cette classe avec votre modèle :")
    print("""
    from etape4_webcam.webcam_live import WebcamLipReader
    
    # Charger le modèle
    model = tf.keras.models.load_model('models/lip_reading_model.h5')
    
    # Créer le lecteur
    reader = WebcamLipReader(
        model=model,
        word_list=['hello', 'goodbye', 'thank', ...],
        sequence_length=20
    )
    
    # Lancer la webcam
    reader.run_webcam()
    """)


if __name__ == "__main__":
    main()
