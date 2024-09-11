from sentence_transformers import SentenceTransformer
import pandas as pd


def vectorize_df(data: pd.DataFrame, 
                 model_name: str = "intfloat/multilingual-e5-large",
                 device: str = "cuda"
                 ) -> pd.DataFrame:
    """
    Vectorize dataframe and return dataframe with "embedding".
    :param data: dataframe with strings.
    :param device: name of the device where the model will be run.
    :return: vectorized data stored in dataframe["embedding"].
    """
    embedding_model = SentenceTransformer(model_name_or_path=model_name,
                                          device=device)
    
    vectorized_data = pd.DataFrame(columns=["embedding"])
    vectorized_data["embedding"] = embedding_model.encode(data).tolist()
    return vectorized_data
