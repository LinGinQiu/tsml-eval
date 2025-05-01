import numpy as np
import warnings
from sklearn.utils import check_random_state
from aeon.transformations.collection import BaseCollectionTransformer
from collections import OrderedDict

# --- SyntheticSampleSelector Voting-based Filtering ---
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
from scipy.special import expit  # sigmoid function
from sklearn.model_selection import train_test_split

class SyntheticSampleSelector:
    """
    A voting-based selection system for filtering synthetic samples using multiple strategies:
    Differential Evolution (DE), Genetic Algorithm (GA), KNN Consistency, and Clustering proximity.

           n_select : int or None
            Number of top scoring synthetic samples to select. If None, use voting threshold.

    """
    def __init__(self, voting_threshold: int | None = None, random_state=None, n_select: int | None = None):
        self.voting_threshold = voting_threshold
        self.random_state = check_random_state(random_state)
        self.n_select = n_select


    def select(self, X_real: np.ndarray, y_real, X_syn: np.ndarray, y_syn, test_size=0.2):
        """
        Parameters
        ----------
        X_real : ndarray, shape (n_real, dim)
            Original minority samples.
        y_real : ndarray, shape (n_real,)
            Original labels (used for training classifier).
        X_syn : ndarray, shape (n_syn, dim)
            Synthetic samples to evaluate.
        y_syn : ndarray, shape (n_syn,)
            Synthetic labels (should be same class).
        test_size : float
            Validation split for model evaluation during GA/DE fitness.

        Returns
        -------
        X_selected, y_selected : filtered synthetic samples
        """
        if X_real.ndim == 3 and X_real.shape[1] == 1:
            X_real = X_real[:, 0, :]
        elif X_real.ndim != 2:
            raise ValueError("Input X_real must be 2D or 3D (n, 1, l)")

        if X_syn.ndim == 3 and X_syn.shape[1] == 1:
            X_syn = X_syn[:, 0, :]
        elif X_syn.ndim != 2:
            raise ValueError("Input X_syn must be 2D or 3D (n, 1, l)")

        scores = np.zeros(len(X_syn), dtype=int)

        # Apply each selector and accumulate votes
        selectors = [
            self._knn_consistency,
            self._cluster_proximity,
            self._genetic_filter,
            self._de_filter
        ]

        for method in selectors:
            selected_mask = method(X_real, y_real, X_syn, y_syn, test_size)
            scores += selected_mask.astype(int)
        if self.n_select is None:
            _, counts = np.unique(y_real, return_counts=True)
            gap = abs(counts[0] - counts[1])
            self.n_select = max(1, gap)

        topk = np.argsort(-scores)[:self.n_select]
        print(topk)
        print(scores[topk])
        print(scores[topk[-1]])
        if self.voting_threshold is not None and scores[topk[-1]] <= self.voting_threshold:
            keep_mask = scores >= self.voting_threshold
            return X_syn[keep_mask], y_syn[keep_mask]
        else:
            return X_syn[topk], y_syn[topk]

    def _knn_consistency(self, X_real, y_real, X_syn, y_syn, test_size):
        X_all = np.concatenate([X_real, X_syn])
        y_all = np.concatenate([y_real, y_syn])
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(X_all)
        neighbors = knn.kneighbors(X_syn, return_distance=False)
        labels = y_all[neighbors]
        consistent = (labels == y_syn[:, None]).sum(axis=1) > 3
        return consistent

    def _cluster_proximity(self, X_real, y_real, X_syn, y_syn, test_size):
        kmeans = KMeans(n_clusters=1, random_state=self.random_state)
        kmeans.fit(X_real)
        distances = np.linalg.norm(X_syn - kmeans.cluster_centers_[0], axis=1)
        threshold = np.percentile(distances, 70)  # keep closer 70%
        return distances <= threshold

    def _genetic_filter(self, X_real, y_real, X_syn, y_syn, test_size):
        X_train, X_val, y_train, y_val = train_test_split(X_real, y_real, test_size=test_size, random_state=self.random_state)
        population_size = 10
        n_gen = 5
        n_samples = len(X_syn)
        pop = self.random_state.randint(0, 2, size=(population_size, n_samples))

        for _ in range(n_gen):
            fitness = []
            for individual in pop:
                X_aug = np.concatenate([X_train, X_syn[individual == 1]])
                y_aug = np.concatenate([y_train, y_syn[individual == 1]])
                clf = RandomForestClassifier(random_state=self.random_state)
                clf.fit(X_aug, y_aug)
                y_pred = clf.predict(X_val)
                fitness.append(f1_score(y_val, y_pred, average='macro'))
            top_idx = np.argsort(fitness)[-population_size//2:]
            offspring = pop[top_idx]
            new_pop = []
            for p in offspring:
                mate = offspring[self.random_state.randint(len(offspring))]
                mask = self.random_state.rand(n_samples) > 0.5
                child = np.where(mask, p, mate)
                if self.random_state.rand() < 0.2:
                    flip = self.random_state.randint(0, n_samples)
                    child[flip] = 1 - child[flip]
                new_pop.append(child)
            pop = np.array(new_pop)

        best_ind = pop[np.argmax(fitness)]
        return best_ind.astype(bool)

    def _de_filter(self, X_real, y_real, X_syn, y_syn, test_size):
        X_train, X_val, y_train, y_val = train_test_split(X_real, y_real, test_size=test_size, random_state=self.random_state)
        n_samples = len(X_syn)
        pop_size = 10
        gens = 5
        F = 0.5
        CR = 0.9
        pop = self.random_state.rand(pop_size, n_samples)

        def evaluate(ind):
            mask = expit(ind) > 0.5
            if mask.sum() == 0:
                return 0
            X_aug = np.concatenate([X_train, X_syn[mask]])
            y_aug = np.concatenate([y_train, y_syn[mask]])
            clf = RandomForestClassifier(random_state=self.random_state)
            clf.fit(X_aug, y_aug)
            y_pred = clf.predict(X_val)
            return f1_score(y_val, y_pred, average='macro')

        for _ in range(gens):
            for i in range(pop_size):
                a, b, c = pop[np.random.choice(pop_size, 3, replace=False)]
                mutant = a + F * (b - c)
                cross = self.random_state.rand(n_samples) < CR
                trial = np.where(cross, mutant, pop[i])
                if evaluate(trial) > evaluate(pop[i]):
                    pop[i] = trial

        best = pop[np.argmax([evaluate(p) for p in pop])]
        return (expit(best) > 0.5)
