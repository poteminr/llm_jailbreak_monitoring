import pickle
import torch
import numpy as np
from sentence_transformers import util, SentenceTransformer

import faiss


class RAGSimInjectionFaiss:
    def __init__(self,
                 model_name_or_path: str="intfloat/multilingual-e5-large",
                 device: str="cuda",
                 path_to_embeddings: str="jailbreak_prompts_embeds.pkl"
                 ) -> None:

        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device)
        
        with open(path_to_embeddings,"rb") as f:
            self.prompts_and_embeddings = pickle.load(f)
            self.embeddings_injection_prompts = np.array(self.prompts_and_embeddings["embeddings"].tolist())

        dimension = self.embeddings_injection_prompts.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(self.embeddings_injection_prompts)
        self.index.add(self.embeddings_injection_prompts)
      
    def find_sim_injection_faiss(self,
                                 query: str,
                                 k: int=10
                                 ) -> tuple[list[str], list[float]]:
        """
        Embeds a new query and returns top k faiss scores and prompts from embeddings.
        :param input: string of user input
        :param k: number of returned prompts
        :return: tuple that contains list of prompts and list of similarity scores
        """

        query_embedding_vector = np.array([self.embedding_model.encode(query)])
        faiss.normalize_L2(query_embedding_vector)
        differencies, indices = self.index.search(query_embedding_vector, k=k)
        sim_injections = self.prompts_and_embeddings.iloc[indices[0]]["prompt"].tolist()
        scores = (1-differencies[0]).tolist()

        return (sim_injections, scores)
