import numpy as np
import faiss

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BaseRetriever: #TODO
    def __init__(self, init_reps: np.ndarray):
        self.index = init_reps

    def add(self, p_reps: np.ndarray):
        self.index = np.concatenate((self.index, p_reps), axis=0)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)


class DotProductRetriever(BaseRetriever): #TODO
    def search(self, q_reps: np.ndarray, k: int):
        # Calculate dot product between query reps and indexed reps
        scores = np.dot(q_reps, self.index.T)

        # Get top-k indices and scores
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_k_scores = np.take_along_axis(scores, indices, axis=1)

        return top_k_scores, indices

class CMERetriever(BaseRetriever): #TODO
    def __init__(self,
                 init_reps: np.ndarray,
                 num_heads = 2, num_layers = 2
                 ):
        super().__init__()

        self.embed_dim = 768 # for coCondenser-marco
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extend_multi_transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
        self.extend_multi_transformerencoder = torch.nn.TransformerEncoder(self.extend_multi_transformerencoderlayer, self.num_layers).to(self.device)

    def search(self, q_reps: np.ndarray, k: int = 1):
        # Calculate dot product between query reps and indexed reps
        #scores = np.dot(q_reps, self.index.T)
        scores = self.extend_multi(q_reps, self.index.T)

        # Get top-k indices and scores
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_k_scores = np.take_along_axis(scores, indices, axis=1)

        return top_k_scores, indices

    def extend_multi(self, q_reps: np.ndarray, p_reps: np.ndarray): # changed to np.ndarray type
        batch_size = q_reps.shape[0]

        # Convert np.ndarrays to PyTorch tensors
        q_reps_tensor = torch.from_numpy(q_reps)
        p_reps_tensor = torch.from_numpy(p_reps)

        # Original tensor operations converted to PyTorch tensor operations
        xs = q_reps_tensor.unsqueeze(dim=1)
        ys = p_reps_tensor.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        input_tensor = torch.cat([xs, ys], dim=1) # concatenate mention and entity embeddings

        # Assuming extend_multi_transformerencoder requires PyTorch tensor
        attention_result = self.extend_multi_transformerencoder(input_tensor)

        # Get score from dot product
        scores_tensor = torch.bmm(attention_result[:, 0, :].unsqueeze(1), attention_result[:, 1:, :].transpose(2, 1))
        scores_tensor = scores_tensor.squeeze(-2)

        # Convert the PyTorch tensor back to a numpy array
        scores = scores_tensor.detach().cpu().numpy()
        return scores


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
