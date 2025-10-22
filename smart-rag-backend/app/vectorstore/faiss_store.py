from __future__ import annotations
from typing import List,Tuple,Literal,Optional
import os
import pickle
from dataclasses import dataclass
import numpy as np
from langchain_core.documents import Document
import faiss

#FAISS import cpu-friendly
# try:
#     import faiss
# except Exception:
#     try:
#         import faiss as faiss
#     except Exception as e:
#         raise ImportError("faiss or faiss-cpu is required. pip install faiss-cpu") from e 

Metric = Literal["cosine","l2"]

def _l2_normalize(mat:np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return mat/norms

@dataclass
class _MemStore:
    texts: List[str]
    metadata: List[dict]

class FaissIndex:
    '''
    Minimal FAISS wrapper with metadata persistence
    '''
    def __init__(self,dimension:int, metric:Metric = "cosine", model_name:str = "") -> None:
        if metric not in ["cosine","l2"]:
            raise ValueError("metric should be either cosine or l2")
        self._dim = int(dimension)
        self._metric: Metric = metric
        self._model_name = model_name

        if self._metric == "cosine":
            self._index = faiss.IndexFlatIP(self._dim)
        elif self._metric == "l2":
            self._index = faiss.IndexFlatL2(self._dim)

        self._store = _MemStore(texts=[],metadata=[])


#---------properties--------------
    @property
    def dimension(self) -> int:
        return self._dim    
        
    @property
    def model_name(self) -> str:
        return self._model_name
        
    @property
    def count(self) -> int:
        return int(self._index.ntotal)        
    

    def set_namespace(self,ns:Optional[str]) -> None:
        self.namespace = ns or "default"

        
#---------core operations-----------------

    def add_documents(self, docs:List[Document],embedder) -> None:
        if not docs:
            return
        
        texts =[d.page_content for d in docs]
        vecs = embedder.encode_texts(texts)

        if vecs.shape[1] != self._dim:
            raise ValueError("vector dimension is not equal to index dim. vector_dim:{vecs.shape[1]}, index dim:{self._dim}")


        if self._metric == "cosine":
            vecs = _l2_normalize(vecs)


        self._index.add(vecs)

        self._store.texts.extend(texts)
        self._store.metadata.extend([dict(d.metadata) for d in docs])



    def search(self, query:str, k:int, embedder) -> List[Document]:
        if self._model_name and getattr(embedder,'model_name',None) != self._model_name:
            raise Exception(
                f"Embedder model {getattr(embedder,'model_name',None)} is not same as index model: {self._model_name}"
            )
        if self.count == 0:
            return []
        
        q = np.asarray(embedder.encode_texts([query]), dtype=np.float32)
        if self._metric == "cosine":
            q = _l2_normalize(q)

        
        distances,indices = self._index.search(q, min(k,self.count))
        idxs = indices[0].tolist()

        results:List[Document] = []
        for i in idxs:
            if i<0:
                continue
            text = self._store.texts[i]
            md = self._store.metadata[i]
            results.append(Document(page_content=text,metadata=md))
        
        return results


#------------- Persistence -----------------------
    def save_local(self,dir_path:str) -> None:
        os.makedirs(dir_path,exist_ok=True)

        faiss.write_index(self._index, os.path.join(dir_path,"index.faiss"))

        payload = {
            "dimension": self._dim,
            "metric":self._metric,
            "model_name":self._model_name,
            "texts":self._store.texts,
            "metadata":self._store.metadata
        }

        with open(os.path.join(dir_path,"meta.pkl"),"wb") as f:
            pickle.dump(payload,f)
        

    
    @classmethod
    def load_local(cls,dirpath:str) -> FaissIndex:
        
        idx_path = os.path.join(dirpath,"index.faiss")
        meta_path = os.path.join(dirpath, "meta.pkl")

        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("index or metadata file not found")
        
        try:
            index = faiss.read_index(idx_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read FAISS index: {e}") from e
        
        with open(meta_path,"rb") as f:
            payload = pickle.load(f)
        
        dim = int(payload["dimension"])
        metric:Metric = payload["metric"]
        model_name = payload.get("model_name","")

        obj = cls(dimension=dim,metric=metric,model_name=model_name)

        obj._index = index

        texts = payload.get('texts',[])
        metadata = payload.get('metadata',[])
        # print(f"index.ntotal is {index.ntotal} of type {type(index.ntotal)}")
        # print(f"len of texts is {len(texts)} of type {type(len(texts))}")
        print(f"len of metadata: {len(metadata)}")
        print(metadata)
        if len(texts) != index.ntotal or len(metadata) != index.ntotal:
            raise RuntimeError("Metadata does not match index vector count")
        
        obj._store = _MemStore(texts=texts,metadata=metadata)
        return obj
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all vectors in the current FAISS index that belong to the given document ID.
        If your FAISS backend keeps a sidecar metadata mapping, it prunes those entries too.
        """
        if not hasattr(self, "_metadatas") or not self._metadatas:
            # No metadata tracking; just reset the index if needed
            return

        try:
            # Collect indices of embeddings NOT matching this doc_id
            keep_indices = [
                i for i, meta in enumerate(self._metadatas)
                if str(meta.get("doc_id")) != str(doc_id)
            ]

            # If all entries are from this doc, just reset everything
            if not keep_indices:
                self.index.reset()
                self._embeddings = []
                self._metadatas = []
                return

            # Otherwise, rebuild index with only the kept embeddings
            kept_embeddings = [self._embeddings[i] for i in keep_indices]
            kept_metadatas = [self._metadatas[i] for i in keep_indices]

            # Recreate FAISS index
            import numpy as np
            self.index.reset()
            if kept_embeddings:
                self.index.add(np.array(kept_embeddings, dtype="float32"))

            self._embeddings = kept_embeddings
            self._metadatas = kept_metadatas

        except Exception as e:
            print(f"[FAISS] delete_by_doc_id({doc_id}) failed: {e}")

