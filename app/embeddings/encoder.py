# app/embeddings/encoder.py
import numpy as np
from typing import List, Optional

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = self._load_model(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for Embedder. "
                "pip install sentence-transformers"
            ) from e
        return SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if len(texts) == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=batch_size or 16,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs)
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        return vecs

# import numpy as np
# from typing import List, Optional


# class Embedder:
#     def __init__(self,model_name: str="sentence-transformers/all-MiniLM-L6-v2") -> None:
#         self.model_name = model_name
#         #Lazy load the model
#         self._model = self._load_model(model_name)
#         self._dimension = self._model.get_sentence_embedding_dimension()
    
#     @staticmethod
#     def _load_model(model_name:str):
#         try:
#             from sentence_transformers import SentenceTransformer
#         except Exception as e:
#             raise ImportError(
#                 "sentence-transformers is required for Embedder."
#                 "pip install sentence-transformers"
#             ) from e
#         return SentenceTransformer(model_name)
    
#     @property
#     def dimension(self) -> int:
#         return int(self._dimension)


#     def encode_texts(self, texts:List[str], batch_size:Optional[int] = None) -> np.ndarray:
#         if not isinstance(texts,List):
#             raise TypeError("texts must be a list of strings")
#         if len(texts) == 0:
#             return np.zeros((0,self._dimension), dtype=np.float32)
#         vecs = self._model.encode(
#             texts,
#             batch_size=batch_size or 16,
#             convert_to_numpy=True,
#             normalize_embeddings=False,
#             show_progress_bar=False
#         )

#         if vecs.dtype != np.ndarray:
#             vecs = vecs.astype(np.float32)
#         return vecs

