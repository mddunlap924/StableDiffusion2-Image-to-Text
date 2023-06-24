import sys
import re
import faiss
import torch
import numpy as np
import polars as pl
import pandas as pd
import gc
from pathlib import Path
import torch.nn.functional as F
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

sys.path.append("../kaggle/input/sentence-transformers-222/sentence-transformers")
from sentence_transformers import SentenceTransformer


def check_string(string: str) -> bool:
    # Checks if the given string contains any character other than alphanumeric characters, comma, dot, hyphen or whitespace
    return bool(re.search(r'[^A-Za-z0-9,.\\-\\s]', string))


def filter_similar_embeddings(df: pd.DataFrame,
                              *,
                              threshold: float=0.9,
                              demo: bool=True,
                              n_neighbors: int=20,
                              batch_size: int=2_000) -> pd.DataFrame:

    # Polars dataframe
    pldf = pl.from_pandas(df.copy())
    # # Select only those images whose width and height fall between 256 and 768 pixels
    # pldf = pldf.filter(pl.col("width").is_between(256, 768) & pl.col("height").is_between(256, 768))

    # Select only those prompts that have five or more words 
    pldf = pldf.filter(pl.col("prompt").str.split(" ").apply(lambda x: len(x)>=3))

    # Select only those prompts that are not blank, NULL, null, or NaN
    pldf = pldf.filter(~pl.col("prompt").str.contains('^(?:\s*|NULL|null|NaN)$'))


    pldf = pldf.filter(pl.col("prompt").apply(check_string))

    #For the purpose of demo/testing, we will reduce the amount of data
    if demo:
        pldf = pldf[:100_000]
    
    # Vectorize using SentenceTransformer
    model = SentenceTransformer("../kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2")
    vector = model.encode(pldf["prompt"].to_numpy(),
                        batch_size=512,
                        show_progress_bar=True,
                        device="cuda",
                        convert_to_tensor=True)
    del model
    _ = gc.collect()
    torch.cuda.empty_cache()
    
    # Create an IndexFlatIP index using the Faiss library
    # The term 'IP' represents the Inner Product, 
    # which is equivalent to cosine similarity as it involves taking the dot product of normalized vectors.
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, 384)

    # Normalize the input vector and add it to the IndexFlatIP 
    index.add(F.normalize(vector).cpu())

    
    similar_vectors = []  # Create an empty list to store similar vectors.
    for i in tqdm(range(0, len(vector), batch_size)):
        # Get the target batch for processing.
        batch_data = vector[i:i + batch_size].cpu()
        # Neighborhood search based on cosine similarity.
        similarities, indices = index.search(batch_data, n_neighbors)
        
        # Extract indexes and similarities of data to be deleted.
        for j in range(similarities.shape[0]):
            close_vectors = indices[j, similarities[j] >= threshold] 
            index_base = i
            # Get only the similar vectors that exclude itself
            close_vectors = close_vectors[close_vectors != index_base + j]  
            similar_vectors.append((index_base + j, close_vectors))
    
    
    remove_idxs = np.unique(np.concatenate([x for _, x in similar_vectors])).tolist()
    # Drop similarity data
    pldf = pldf.with_columns(pl.Series(values=list(range(len(pldf))), name="index"))
    pldf = pldf.filter(~pl.col("index").is_in(remove_idxs))
    df = pldf.to_pandas()
    df = df.drop(columns=['index']).reset_index(drop=True)
    
    del vector
    _ = gc.collect()
    torch.cuda.empty_cache()
    
    return df


def is_below_similarity_threshold(embedding, filtered_indices, normalized_embeddings, threshold=0.95):
    if len(filtered_indices) == 0:
        return True

    # Calculate cosine similarities with existing filtered embeddings
    similarities = torch.matmul(embedding, normalized_embeddings[filtered_indices].t())

    # Check if all similarities are below the threshold
    return torch.all(similarities < threshold)


def filter_embeddings_torch(df: pd.DataFrame,
                              *,
                              threshold: float=0.9,
                              demo: bool=True,
                              n_neighbors: int=20,
                              batch_size: int=2_000) -> pd.DataFrame:

    # Polars dataframe
    pldf = pl.from_pandas(df.copy())
    # # Select only those images whose width and height fall between 256 and 768 pixels
    # pldf = pldf.filter(pl.col("width").is_between(256, 768) & pl.col("height").is_between(256, 768))

    # Select only those prompts that have five or more words 
    pldf = pldf.filter(pl.col("prompt").str.split(" ").apply(lambda x: len(x)>=3))

    # Select only those prompts that are not blank, NULL, null, or NaN
    pldf = pldf.filter(~pl.col("prompt").str.contains('^(?:\s*|NULL|null|NaN)$'))


    pldf = pldf.filter(pl.col("prompt").apply(check_string))

    #For the purpose of demo/testing, we will reduce the amount of data
    if demo:
        pldf = pldf[:10_000]
    
    # Vectorize using SentenceTransformer
    model = SentenceTransformer("../kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2")
    vector = model.encode(pldf["prompt"].to_numpy(),
                        batch_size=512,
                        show_progress_bar=True,
                        device="cuda",
                        convert_to_tensor=True)
    
    # Load your embeddings as a PyTorch tensor and move it to the GPU
    # embeddings = torch.tensor(vector, dtype=torch.float32).cuda()

    # Normalize the embeddings so that each has a unit length (L2 norm = 1)
    normalized_embeddings = vector / torch.norm(vector, dim=1, keepdim=True)

    # Initialize the filtered embeddings indices list
    filtered_indices = []

    # Iterate through the embeddings, checking their similarity with the filtered embeddings
    for i, embedding in tqdm(enumerate(normalized_embeddings), total=len(normalized_embeddings)):
        below_sim_thres = is_below_similarity_threshold(embedding,
                                                        filtered_indices,
                                                        normalized_embeddings,
                                                        threshold=threshold)
        if below_sim_thres:
            filtered_indices.append(i)

    # Convert the filtered_indices list to a tensor
    filtered_indices = torch.tensor(filtered_indices, dtype=torch.long, device=vector.device)
    df = df.loc[filtered_indices.tolist()].reset_index(drop=True)

    # # Get the filtered embeddings using the indices
    # filtered_embeddings = normalized_embeddings[filtered_indices]
    
    return df


def filt_similar_embeddings(df: pd.DataFrame,
                            *,
                            threshold: float=0.9,
                            demo: bool=True,
                            n_neighbors: int=1_000,
                            batch_size: int=2_000) -> pd.DataFrame:

    # Polars dataframe
    pldf = pl.from_pandas(df.copy())
    # # Select only those images whose width and height fall between 256 and 768 pixels
    # pldf = pldf.filter(pl.col("width").is_between(256, 768) & pl.col("height").is_between(256, 768))

    # Select only those prompts that have five or more words 
    pldf = pldf.filter(pl.col("prompt").str.split(" ").apply(lambda x: len(x)>=3))

    # Select only those prompts that are not blank, NULL, null, or NaN
    pldf = pldf.filter(~pl.col("prompt").str.contains('^(?:\s*|NULL|null|NaN)$'))


    pldf = pldf.filter(pl.col("prompt").apply(check_string))
    
    # Vectorize using SentenceTransformer
    model = SentenceTransformer("../kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2")
    vector = model.encode(pldf["prompt"].to_numpy(),
                        batch_size=512,
                        show_progress_bar=True,
                        device="cuda",
                        convert_to_tensor=True)
    del model
    _ = gc.collect()
    torch.cuda.empty_cache()
    
    # Create an IndexFlatIP index using the Faiss library
    # The term 'IP' represents the Inner Product, 
    # which is equivalent to cosine similarity as it involves taking the dot product of normalized vectors.
    resources = faiss.StandardGpuResources()
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(384), 384, 5, faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)

    # Normalize the input vector and add it to the IndexFlatIP 
    gpu_index.train(F.normalize(vector).cpu().numpy())
    gpu_index.add(F.normalize(vector).cpu().numpy())

    
    similar_vectors = []  # Create an empty list to store similar vectors.
    for i in tqdm(range(0, len(vector), batch_size)):
        # Get the target batch for processing.
        batch_data = vector.cpu().numpy()[i:i + batch_size]
        # Neighborhood search based on cosine similarity.
        similarities, indices = gpu_index.search(batch_data, n_neighbors)
        
        # Extract indexes and similarities of data to be deleted.
        for j in range(similarities.shape[0]):
            close_vectors = indices[j, similarities[j] >= threshold] 
            index_base = i
            # Get only the similar vectors that exclude itself
            close_vectors = close_vectors[close_vectors != index_base + j]  
            similar_vectors.append((index_base + j, close_vectors))
    
    # Drop similarity data
    pldf = pldf.with_columns(pl.Series(values=list(range(len(pldf))), name="index"))
    pldf = pldf.filter(~pl.col("index").is_in(np.unique(np.concatenate([x for _, x in similar_vectors])).tolist()))
    df = pldf.to_pandas()
    df = df.drop(columns=['index']).reset_index(drop=True)
    
    del vector
    _ = gc.collect()
    torch.cuda.empty_cache()
    
    return df