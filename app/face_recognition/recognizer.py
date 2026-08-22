"""
Face Recognizer Module using ArcFace Feature Embeddings and Cosine Metric.

Key AI/ML Concepts:
- Metric Learning with ArcFace: Maps face identity to hyperspherical angular space.
- Cosine Similarity: Measures angular proximity between normalized feature vectors.
  Formula: Sim(u, v) = (u . v) / (||u|| * ||v||)
  Range: [-1.0, 1.0]. A higher score indicates stronger biometric match.
- Vectorized Matrix Comparison: Computes dot product between the query vector (1, 512)
  and the entire gallery matrix (N, 512) in a single BLAS-accelerated operation.
- Threshold Decision Boundary: Configurable acceptance threshold (default: 0.45).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import app.config as config
from app.face_recognition.embedding_store import EmbeddingStore


class FaceRecognizer:
    """
    Performs real-time face matching against registered student biometric centroids.
    """

    def __init__(self, 
                 store: Optional[EmbeddingStore] = None,
                 threshold: float = config.RECOGNITION_THRESHOLD):
        """
        Initialize the Face Recognizer.

        Args:
            store: Instance of EmbeddingStore (creates new if None)
            threshold: Cosine similarity cutoff (e.g. 0.45)
        """
        self.store = store if store is not None else EmbeddingStore()
        self.threshold = threshold
        self._gallery_ids: List[str] = []
        self._gallery_matrix: Optional[np.ndarray] = None
        self._rebuild_gallery_cache()

    def reload(self):
        """Reloads stored embeddings from disk and refreshes in-memory matrix."""
        self.store.load()
        self._rebuild_gallery_cache()

    def _rebuild_gallery_cache(self):
        """
        Builds a contiguous (N, 512) numpy matrix for fast vectorized matrix multiplication.
        """
        data = self.store.get_all_embeddings()
        self._gallery_ids = []
        vectors = []

        for sid, info in data.items():
            emb = info.get("embedding")
            if emb is not None:
                # Ensure normalized float32
                norm_emb = EmbeddingStore.normalize_embedding(np.asarray(emb, dtype=np.float32))
                self._gallery_ids.append(sid)
                vectors.append(norm_emb)

        if vectors:
            self._gallery_matrix = np.vstack(vectors)  # Shape: (N, 512)
        else:
            self._gallery_matrix = None

        print(f"[FaceRecognizer] Gallery cache updated. Active registered identities: {len(self._gallery_ids)}")

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Computes cosine similarity between two feature vectors.
        Sim(A, B) = dot(A, B) / (norm(A) * norm(B))
        """
        a = vec_a.flatten().astype(np.float32)
        b = vec_b.flatten().astype(np.float32)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return similarity

    def recognize_face(self, probe_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Matches a single detected face embedding against the registered student gallery.

        Args:
            probe_embedding: 512-D ArcFace embedding vector

        Returns:
            Dictionary with:
            - is_recognized: bool
            - student_id: str ("Unknown" if below threshold)
            - name: str ("Unknown" if below threshold)
            - roll_number: str
            - confidence: float (cosine similarity score)
            - threshold: float (applied decision threshold)
        """
        if probe_embedding is None or self._gallery_matrix is None or len(self._gallery_ids) == 0:
            return {
                "is_recognized": False,
                "student_id": "Unknown",
                "name": "Unknown",
                "roll_number": "",
                "confidence": 0.0,
                "threshold": self.threshold
            }

        # Normalize probe vector
        probe_norm = EmbeddingStore.normalize_embedding(np.asarray(probe_embedding, dtype=np.float32)).reshape(1, -1)

        # Vectorized Dot Product against (N, 512) gallery
        # Since both probe and gallery vectors are unit normalized, dot product equals cosine similarity
        similarities = np.dot(self._gallery_matrix, probe_norm.T).flatten()  # Shape: (N,)

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_student_id = self._gallery_ids[best_idx]

        # Apply Decision Boundary
        if best_score >= self.threshold:
            student_info = self.store.get_student(best_student_id)
            return {
                "is_recognized": True,
                "student_id": best_student_id,
                "name": student_info.get("name", "Unknown") if student_info else "Unknown",
                "roll_number": student_info.get("roll_number", "") if student_info else "",
                "confidence": best_score,
                "threshold": self.threshold
            }
        else:
            return {
                "is_recognized": False,
                "student_id": "Unknown",
                "name": "Unknown",
                "roll_number": "",
                "confidence": best_score,
                "threshold": self.threshold
            }

    def recognize_faces(self, faces: list) -> List[Dict[str, Any]]:
        """
        Multi-face recognition for all detected face objects in a video frame.

        Args:
            faces: List of InsightFace Face objects

        Returns:
            List of recognition result dictionaries corresponding to each face.
        """
        results = []
        for face in faces:
            if hasattr(face, 'embedding') and face.embedding is not None:
                rec_res = self.recognize_face(face.embedding)
            else:
                rec_res = {
                    "is_recognized": False,
                    "student_id": "Unknown",
                    "name": "Unknown",
                    "roll_number": "",
                    "confidence": 0.0,
                    "threshold": self.threshold
                }
            rec_res["face"] = face
            results.append(rec_res)
        return results
