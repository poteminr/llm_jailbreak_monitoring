import pickle
import torch
import numpy as np
import faiss
from sentence_transformers import util, SentenceTransformer


class InjectionFinder:
    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-large",
        path_to_embeddings: str = "jailbreak_prompts_embeds.pkl",
        device: str = "cuda"
    ) -> None:
        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device
        )
        
        with open(path_to_embeddings, "rb") as f:
            self.prompts_and_embeddings = pickle.load(f)
            self.embeddings_injection_prompts = torch.tensor(np.array(self.prompts_and_embeddings["embeddings"].tolist()))
            self.embeddings_injection_prompts = self.embeddings_injection_prompts.to(device)
      
    def find_sim_injection(
        self,
        query: str,
        top_k: int = 10
    ) -> tuple[list[str], list[float]]:
        """
        Embeds a new query and returns top k scores and prompts from embeddings.
        :param query: string of user input
        :param k: number of returned prompts
        :return: tuple that contains list of prompts and list of similarity scores
        """
        
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dot_scores = util.dot_score(query_embedding, self.embeddings_injection_prompts)[0]
        scores, indices = torch.topk(input=dot_scores, k=top_k)
        sim_injections = self.prompts_and_embeddings.iloc[indices.cpu()]["prompt"]

        return sim_injections.tolist(), scores.tolist()


class InjectionFinderFaiss:
    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-large",
        path_to_embeddings: str = "jailbreak_prompts_embeds.pkl",
        device: str = "cuda"
    ) -> None:
        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device
        )
        
        with open(path_to_embeddings, "rb") as f:
            self.prompts_and_embeddings = pickle.load(f)
            self.embeddings_injection_prompts = np.array(self.prompts_and_embeddings["embeddings"].tolist())

        dimension = self.embeddings_injection_prompts.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(self.embeddings_injection_prompts)
        self.index.add(self.embeddings_injection_prompts)
      
    def find_sim_injection(
        self,
        query: str,
        top_k: int = 10
    ) -> tuple[list[str], list[float]]:
        """
        Embeds a new query and returns top k faiss scores and prompts from embeddings.
        :param input: string of user input
        :param k: number of returned prompts
        :return: tuple that contains list of prompts and list of similarity scores
        """

        query_embedding_vector = np.array([self.embedding_model.encode(query)])
        faiss.normalize_L2(query_embedding_vector)
        differencies, indices = self.index.search(query_embedding_vector, k=top_k)
        sim_injections = self.prompts_and_embeddings.iloc[indices[0]]["prompt"].tolist()
        scores = (1-differencies[0]).tolist()

        return sim_injections, scores