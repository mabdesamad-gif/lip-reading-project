"""
ÉTAPE 2 : Extraction des Lèvres
Détection et extraction de la région des lèvres à partir de vidéos
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle


class LipExtractor:
    """
    Classe pour extraire les lèvres d'une vidéo
    """
    
    def __init__(self, target_size=(96, 48)):
        """
        Initialiser le détecteur de visage et les paramètres
        
        Args:
            target_size: Taille du crop des lèvres (hauteur, largeur)
        """
        self.target_size = target_size
        
        # Initialiser MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Indices des lèvres dans FaceMesh (468 landmarks)
        # Lèvres extérieures et intérieures
        self.lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            178, 200, 199, 175, 171, 77, 90, 180, 85, 16,
        ]
    
    def get_lip_region(self, frame, landmarks):
        """
        Extraire la région des lèvres d'une frame
        
        Args:
            frame: Frame vidéo (numpy array)
            landmarks: Landmarks détectés par MediaPipe
        
        Returns:
            lip_crop: Région des lèvres crop et resizée
            success: Booléen indiquant le succès de l'extraction
        """
        try:
            h, w, c = frame.shape
            
            # Obtenir les coordonnées des lèvres
            lip_points = np.array([
                [landmarks[idx].x * w, landmarks[idx].y * h]
                for idx in self.lip_indices
            ], dtype=np.int32)
            
            # Calculer le bounding box des lèvres
            x_min, y_min = lip_points.min(axis=0)
            x_max, y_max = lip_points.max(axis=0)
            
            # Ajouter un padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop la région des lèvres
            lip_region = frame[y_min:y_max, x_min:x_max]
            
            if lip_region.size == 0:
                return None, False
            
            # Redimensionner à la taille cible
            lip_crop = cv2.resize(lip_region, self.target_size)
            
            # Normaliser (0-1)
            lip_crop = lip_crop.astype(np.float32) / 255.0
            
            return lip_crop, True
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des lèvres : {e}")
            return None, False
    
    def extract_sequence(self, video_path, sequence_length=20):
        """
        Extraire une séquence de frames des lèvres d'une vidéo
        
        Args:
            video_path: Chemin vers la vidéo
            sequence_length: Nombre de frames par séquence (défaut 20)
        
        Returns:
            sequences: Liste de séquences de lèvres
            success: Booléen indiquant le succès
        """
        sequences = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo : {video_path}")
            return [], False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Vidéo : {frame_count} frames, {fps:.2f} FPS")
        
        current_sequence = []
        frame_idx = 0
        
        with tqdm(total=frame_count, desc="Extraction des lèvres") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Redimensionner la frame pour la détection (plus rapide)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                
                # Détecter les landmarks
                results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Extraire les lèvres
                    lip_crop, success = self.get_lip_region(frame, landmarks)
                    
                    if success:
                        current_sequence.append(lip_crop)
                        
                        # Quand on a une séquence complète
                        if len(current_sequence) == sequence_length:
                            sequences.append(np.array(current_sequence))
                            current_sequence = []
                else:
                    # Reset si pas de visage détecté
                    current_sequence = []
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # Ajouter la dernière séquence incomplète si elle est suffisamment longue
        if len(current_sequence) >= sequence_length // 2:
            # Padding avec la dernière frame
            while len(current_sequence) < sequence_length:
                current_sequence.append(current_sequence[-1])
            sequences.append(np.array(current_sequence))
        
        return sequences, len(sequences) > 0
    
    def extract_and_save(self, video_path, output_dir, label=""):
        """
        Extraire et sauvegarder les séquences de lèvres
        
        Args:
            video_path: Chemin vers la vidéo
            output_dir: Répertoire de sortie
            label: Label de la vidéo (optionnel)
        
        Returns:
            success: Booléen indiquant le succès
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Traitement : {Path(video_path).name}")
        
        sequences, success = self.extract_sequence(video_path)
        
        if not success:
            print(f"⚠️  Aucune séquence extraite")
            return False
        
        print(f"✅ {len(sequences)} séquence(s) extraite(s)")
        
        # Sauvegarder les séquences
        base_name = Path(video_path).stem
        
        for i, seq in enumerate(sequences):
            output_file = output_dir / f"{base_name}_seq_{i}.npy"
            np.save(str(output_file), seq)
            
            # Sauvegarder aussi les métadonnées
            metadata = {
                'source': str(video_path),
                'label': label,
                'sequence_idx': i,
                'shape': seq.shape,
                'dtype': str(seq.dtype)
            }
            
            metadata_file = output_dir / f"{base_name}_seq_{i}_meta.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"💾 Séquences sauvegardées dans : {output_dir}")
        return True


def main():
    """
    Démonstration d'utilisation
    """
    print("\n" + "="*60)
    print("📖 ÉTAPE 2 : Extraction des Lèvres")
    print("="*60)
    
    # Initialiser l'extracteur
    extractor = LipExtractor(target_size=(96, 48))
    
    print("\n✨ Extracteur de lèvres initialisé !")
    print(f"   Taille cible : {extractor.target_size}")
    print(f"   Landmarks utilisés : {len(extractor.lip_indices)}")
    
    # Exemple d'utilisation avec une vidéo test
    print("\n📝 Exemple d'utilisation :")
    print("""
    # Extraire d'une seule vidéo
    extractor.extract_and_save(
        video_path='path/to/video.mp4',
        output_dir='data/processed/lips',
        label='test_word'
    )
    
    # Ou extraire une séquence directement
    sequences, success = extractor.extract_sequence('path/to/video.mp4')
    if success:
        print(f'Extraits : {len(sequences)} séquences')
        for seq in sequences:
            print(f'  Shape: {seq.shape}')
    """)


if __name__ == "__main__":
    main()
