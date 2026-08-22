"""
Face Embedding Storage and Feature Management Module.

Key AI/ML Concepts:
- ArcFace Deep Embeddings: Represents facial biometric features in a 512-dimensional hypersphere.
- Multi-Sample Ensembling: Averages embeddings from multiple angles (Front, Left, Right, Up, Down)
  to construct a robust centroid prototype for each identity.
- L2 Normalization: Scales embedding vectors to unit length ||v|| = 1 such that Cosine Similarity
  is equivalent to the Euclidean Dot Product (dot(u, v)).
- Local Persistence: Stores serialized biometric matrices in PKL format and metadata in JSON.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import app.config as config


class EmbeddingStore:
    """
    Manages registration, multi-sample averaging, L2 normalization,
    and local persistence of facial feature embeddings.
    """

    def __init__(self, 
                 embeddings_path: Path = config.EMBEDDINGS_FILE,
                 meta_path: Path = config.STUDENT_META_FILE):
        """
        Initialize the embedding store.

        Args:
            embeddings_path: Path to the .pkl file storing embedding vectors
            meta_path: Path to the .json file storing student metadata
        """
        self.embeddings_path = Path(embeddings_path)
        self.meta_path = Path(meta_path)
        
        # Internal dictionary storage:
        # {
        #    student_id: {
        #        "roll_number": str,
        #        "name": str,
        #        "embedding": np.ndarray (512,),
        #        "sample_count": int,
        #        "registered_at": str
        #    }
        # }
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2 Normalizes a 1D or 2D embedding vector onto the unit hypersphere.
        Formula: v_norm = v / ||v||_2
        """
        norm = np.linalg.norm(embedding)
        if norm == 0 or np.isnan(norm):
            return embedding
        return (embedding / norm).astype(np.float32)

    @classmethod
    def compute_average_embedding(cls, sample_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Computes the robust centroid embedding across multiple face samples.
        1. Averages raw feature vectors across all valid captures.
        2. Applies L2 normalization to produce a unit vector representation.

        Args:
            sample_embeddings: List of 512-D numpy arrays

        Returns:
            Normalized 512-D numpy array
        """
        if not sample_embeddings:
            raise ValueError("Cannot compute average embedding from empty sample list.")

        # Ensure all are 1D float32 arrays
        clean_samples = [np.asarray(s, dtype=np.float32).flatten() for s in sample_embeddings]
        
        # Compute element-wise mean vector
        mean_vector = np.mean(clean_samples, axis=0)
        
        # L2 Normalize
        normalized_vector = cls.normalize_embedding(mean_vector)
        return normalized_vector

    def add_student(self, 
                    student_id: str, 
                    roll_number: str, 
                    name: str, 
                    sample_embeddings: List[np.ndarray],
                    registered_at: Optional[str] = None) -> np.ndarray:
        """
        Registers or updates a student in the database.
        
        Args:
            student_id: Unique identifier (e.g. '101')
            roll_number: Academic roll number (e.g. '23CS001')
            name: Full name of the student
            sample_embeddings: List of 512-D embeddings captured from camera
            registered_at: Timestamp string (ISO format or date)
            
        Returns:
            The final normalized representative embedding vector.
        """
        student_id = str(student_id).strip()
        roll_number = str(roll_number).strip()
        name = str(name).strip()

        if not student_id or not name:
            raise ValueError("student_id and name cannot be empty.")

        final_embedding = self.compute_average_embedding(sample_embeddings)

        self.data[student_id] = {
            "student_id": student_id,
            "roll_number": roll_number,
            "name": name,
            "embedding": final_embedding,
            "sample_count": len(sample_embeddings),
            "registered_at": registered_at or ""
        }

        self.save()
        print(f"[EmbeddingStore] Successfully stored identity: ID={student_id}, Name='{name}', Samples={len(sample_embeddings)}")
        return final_embedding

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve student record by ID."""
        return self.data.get(str(student_id).strip())

    def delete_student(self, student_id: str) -> bool:
        """
        Deletes a student from the store.
        Returns True if deleted, False if not found.
        """
        student_id = str(student_id).strip()
        if student_id in self.data:
            del self.data[student_id]
            self.save()
            print(f"[EmbeddingStore] Deleted student ID: {student_id}")
            return True
        return False

    def list_students(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered student summary dictionaries."""
        results = []
        for sid, info in self.data.items():
            results.append({
                "student_id": sid,
                "roll_number": info.get("roll_number", ""),
                "name": info.get("name", ""),
                "sample_count": info.get("sample_count", 0),
                "registered_at": info.get("registered_at", "")
            })
        return results

    def get_all_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Returns the complete in-memory dictionary of embeddings and metadata."""
        return self.data

    def save(self):
        """
        Persists the embeddings to a binary PKL file and metadata to a readable JSON file.
        """
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save binary Pickle (contains numpy arrays)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 2. Save readable JSON metadata (without binary arrays)
        meta_dict = {}
        for sid, info in self.data.items():
            meta_dict[sid] = {
                "student_id": info["student_id"],
                "roll_number": info["roll_number"],
                "name": info["name"],
                "sample_count": info["sample_count"],
                "registered_at": info.get("registered_at", ""),
                "embedding_dim": len(info["embedding"]) if "embedding" in info else 0
            }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    def load(self):
        """
        Loads embeddings and metadata from disk if files exist.
        """
        if self.embeddings_path.exists():
            try:
                with open(self.embeddings_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict):
                        self.data = loaded
                        print(f"[EmbeddingStore] Loaded {len(self.data)} registered student(s) from {self.embeddings_path.name}")
                    else:
                        print(f"[EmbeddingStore] Warning: Invalid file format in {self.embeddings_path}. Starting empty.")
                        self.data = {}
            except Exception as e:
                print(f"[EmbeddingStore] Error loading {self.embeddings_path}: {e}. Starting fresh.")
                self.data = {}
        else:
            self.data = {}
            print("[EmbeddingStore] No existing database found. Initialized empty store.")
