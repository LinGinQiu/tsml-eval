import numpy as np
import warnings
from sklearn.utils import check_random_state
from collections import OrderedDict

# --- SyntheticSampleSelector Voting-based Filtering ---
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist  # For distance calculations
from scipy.special import expit  # sigmoid function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # For scaling distances if needed


class SyntheticSampleSelector:
    """
    A voting-based selection system for filtering synthetic samples using multiple strategies:
    Differential Evolution (DE), Genetic Algorithm (GA), KNN Consistency, and Clustering proximity.

    Parameters
    ----------
    voting_threshold : int or None, default=None
        Number of votes a synthetic sample needs to be selected. If None, n_select is used.
    random_state : int or None, default=None
        Seed for random number generator.
    n_select : int or None, default=None
        Number of top scoring synthetic samples to select. If None, it's determined based on
        the class imbalance in X_real, aiming to balance the classes.
    diversity_strength : float, default=0.5
        Controls the balance between score and diversity.
        0.0 means only score, 1.0 means only diversity.
        A value between 0 and 1.
    diversity_metric : str, default='euclidean'
        Distance metric to use for diversity calculation. e.g., 'euclidean', 'cosine'.
    minority_class_label : int or str, default=1
        The label of the minority class in y_real and y_syn.
        This is crucial for calculating the class imbalance for n_select.
    """

    def __init__(self, voting_threshold: int | None = None, random_state=None, n_select: int | None = None,
                 diversity_strength: float = 0.5, diversity_metric: str = 'euclidean', minority_class_label=None):
        self.voting_threshold = voting_threshold
        self.random_state = check_random_state(random_state)
        self.n_select = n_select
        if not (0.0 <= diversity_strength <= 1.0):
            raise ValueError("diversity_strength must be between 0.0 and 1.0")
        self.diversity_strength = diversity_strength
        self.diversity_metric = diversity_metric
        self.minority_class_label = minority_class_label

    def select(self, X_real: np.ndarray, y_real, X_syn: np.ndarray, y_syn, test_size=0.2):
        """
        Parameters
        ----------
        X_real : ndarray, shape (n_real, 1, n_timestamps) or (n_real, n_timestamps)
            Original dataset samples (majority and minority classes).
        y_real : ndarray, shape (n_real,)
            Original labels corresponding to X_real.
        X_syn : ndarray, shape (n_syn, 1, n_timestamps) or (n_syn, n_timestamps)
            Synthetic minority samples to evaluate.
        y_syn : ndarray, shape (n_syn,)
            Synthetic labels (should all be minority_class_label).
        test_size : float
            Validation split for model evaluation during GA/DE fitness.

        Returns
        -------
        X_selected, y_selected : filtered synthetic samples (3D ndarrays)
        """
        # --- Input validation and reshaping to 2D for processing ---
        if X_real.ndim == 2:
            X_real_2d = X_real
        elif X_real.ndim == 3 and X_real.shape[1] == 1:
            X_real_2d = X_real[:, 0, :]
        else:
            raise ValueError("X_real must be 3D with shape (n_real, 1, n_timestamps) or 2D with shape (n_real, n_timestamps)")

        if X_syn.ndim == 2:
            X_syn_2d = X_syn
        elif X_syn.ndim == 3 and X_syn.shape[1] == 1:
            X_syn_2d = X_syn[:, 0, :]
        else:
            raise ValueError("X_syn must be 3D with shape (n_syn, 1, n_timestamps) or 2D with shape (n_syn, n_timestamps)")

        if self.minority_class_label is None:
            # 获取所有标签及其计数
            labels, counts = np.unique(y_real, return_counts=True)

            # 找到数量最少的那个标签
            self.minority_class_label = labels[np.argmin(counts)]
        # Ensure y_syn labels are consistent with minority_class_label
        if not np.all(y_syn == self.minority_class_label):
            warnings.warn("y_syn contains labels different from minority_class_label. "
                          "Synthetic samples are expected to be only of the minority class.")

        scores = np.zeros(len(X_syn_2d), dtype=int)

        # Apply each selector and accumulate votes
        selectors = [
            self._nan_consistency,
            self._cluster_proximity,
            self._genetic_filter,
            self._de_filter
        ]

        # Pass 2D arrays to selectors
        for method in selectors:
            selected_mask = method(X_real_2d, y_real, X_syn_2d, y_syn, test_size)
            scores += selected_mask.astype(int)

        # --- Determine n_select if not provided ---
        if self.n_select is None:
            unique_classes, counts = np.unique(y_real, return_counts=True)
            if len(unique_classes) < 2:
                raise ValueError("y_real must contain at least two classes to determine imbalance for n_select.")

            majority_class_label = unique_classes[np.argmax(counts)]
            minority_class_label = unique_classes[np.argmin(counts)]

            # Ensure self.minority_class_label matches what was found
            if self.minority_class_label != minority_class_label:
                warnings.warn(f"Provided minority_class_label ({self.minority_class_label}) "
                              f"does not match detected minority class ({minority_class_label}) from y_real. "
                              "Using detected minority class for n_select calculation.")
                self.minority_class_label = minority_class_label

            n_minority_real = counts[unique_classes == self.minority_class_label][0]
            n_majority_real = counts[unique_classes == majority_class_label][0]

            self.n_select = max(1, n_majority_real - n_minority_real)  # Aim to balance classes

        # Handle case where no synthetic samples are provided
        if len(X_syn_2d) == 0:
            return np.array([]).reshape(0, 1, X_real.shape[2]), np.array([])  # Return empty 3D arrays

        # --- Diversity-aware Selection ---
        normalized_scores = scores / len(selectors)  # Normalize between 0 and 1
        n_syn = len(X_syn_2d)

        chosen_indices = []
        available_indices = list(range(n_syn))
        # 1. Select the first sample based purely on highest score
        if n_syn > 0:
            first_idx = np.argmax(normalized_scores)
            chosen_indices.append(first_idx)
            available_indices.remove(first_idx)

        # 2. Iteratively select remaining samples based on combined score and diversity
        while len(chosen_indices) < self.n_select and available_indices:
            best_candidate_idx = -1
            max_combined_score = -np.inf

            current_chosen_X = X_syn_2d[chosen_indices]

            # Calculate distances from all available samples to the already chosen ones
            min_distances_to_chosen = np.min(
                cdist(X_syn_2d[available_indices], current_chosen_X, metric=self.diversity_metric), axis=1)

            # Normalize distances for combination with scores.
            # Scale distances to 0-1 range to be comparable with normalized_scores.
            if len(min_distances_to_chosen) > 0 and np.max(min_distances_to_chosen) > 0:
                normalized_distances = min_distances_to_chosen / np.max(min_distances_to_chosen)
            else:
                normalized_distances = np.zeros_like(min_distances_to_chosen)

            for i, original_idx in enumerate(available_indices):
                score_component = normalized_scores[original_idx]
                diversity_component = normalized_distances[i]

                # Combine score and diversity using the diversity_strength weight
                combined_score = (1 - self.diversity_strength) * score_component + \
                                 self.diversity_strength * diversity_component

                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_candidate_idx = original_idx

            if best_candidate_idx != -1:
                chosen_indices.append(best_candidate_idx)
                available_indices.remove(best_candidate_idx)
            else:
                # This should ideally not happen if available_indices is not empty.
                # Break to prevent infinite loop if no candidate improves.
                break

        # 3. If after diversity selection, we still have fewer than n_select,
        #    fill up with remaining highest-score samples that weren't picked for diversity.
        if len(chosen_indices) < self.n_select:
            remaining_scores = np.copy(normalized_scores)
            remaining_scores[chosen_indices] = -np.inf  # Exclude already chosen
            additional_needed = self.n_select - len(chosen_indices)
            if additional_needed > 0:
                # Get top 'additional_needed' from remaining
                # Using argsort on negative scores for descending order
                top_remaining_indices = np.argsort(remaining_scores)[-additional_needed:]
                # Add unique indices to chosen_indices
                for idx in top_remaining_indices:
                    if idx not in chosen_indices:
                        chosen_indices.append(idx)

        # 4. Apply voting_threshold as a final filter IF specified
        # If voting_threshold is set, it overrides n_select in terms of final count if it results in fewer samples.
        if self.voting_threshold is not None:
            # Filter the already diversity-selected samples based on their raw scores
            temp_chosen_indices = np.array(chosen_indices)  # Convert to array for boolean indexing
            final_mask_by_threshold = (scores[temp_chosen_indices] >= self.voting_threshold)
            chosen_indices = temp_chosen_indices[final_mask_by_threshold].tolist()

        # Convert to numpy array for indexing
        chosen_indices = np.array(chosen_indices)

        # --- Reshape output to 3D ---
        X_selected = X_syn[chosen_indices]
        y_selected = y_syn[chosen_indices]

        return X_selected, y_selected

    def _nan_consistency(self, X_real_2d, y_real, X_syn_2d, y_syn, test_size):
        from sklearn.neighbors import NearestNeighbors

        # Concatenate 2D versions for KNN
        X_all_2d = np.concatenate([X_real_2d, X_syn_2d])
        y_all = np.concatenate([y_real, y_syn])
        n_total = X_all_2d.shape[0]

        if n_total < 2:  # Cannot find neighbors if less than 2 samples
            return np.zeros(len(X_syn_2d), dtype=bool)

        max_r = 20  # Max neighbors to check for reciprocal
        nn_found = False
        for r in range(1, max_r + 1):
            n_neighbors_val = min(r + 1, n_total)  # Ensure n_neighbors doesn't exceed available samples
            nn = NearestNeighbors(n_neighbors=n_neighbors_val).fit(X_all_2d)
            knn_indices = nn.kneighbors(X_all_2d, return_distance=False)

            # Take up to r neighbors, excluding self (index 0)
            knn = knn_indices[:, 1:n_neighbors_val]

            natural_neighbors = [set() for _ in range(n_total)]
            all_have_natural_neighbors = True

            for i in range(n_total):
                found_reciprocal = False
                for j in knn[i]:
                    if i in knn_indices[j, 1:n_neighbors_val]:  # Check if i is one of j's non-self neighbors
                        natural_neighbors[i].add(j)
                        found_reciprocal = True
                if not found_reciprocal:
                    all_have_natural_neighbors = False
                    break  # Break if any sample doesn't have a reciprocal neighbor at this 'r'

            if all_have_natural_neighbors:
                nn_found = True
                break

        if not nn_found:
            warnings.warn(f"Could not find reciprocal nearest neighbors for all samples within r={max_r}. "
                          "KNN consistency filter might be less effective or disabled.")
            return np.zeros(len(X_syn_2d), dtype=bool)  # If no natural neighbors found, consider all inconsistent.

        consistent_mask = np.ones(len(X_syn_2d), dtype=bool)
        # Iterate only over synthetic samples (indices from len(X_real_2d) to n_total-1)
        for i in range(len(X_real_2d), n_total):
            xi_label = y_all[i]
            is_consistent = True

            # If a synthetic sample has no natural neighbors (e.g., in very sparse data)
            # consider it inconsistent.
            if not natural_neighbors[i]:
                is_consistent = False
            else:
                for j in natural_neighbors[i]:
                    if y_all[j] != xi_label:
                        is_consistent = False
                        break  # Found an inconsistent neighbor, no need to check further
            consistent_mask[i - len(X_real_2d)] = is_consistent
        return consistent_mask

    def _cluster_proximity(self, X_real_2d, y_real, X_syn_2d, y_syn, test_size):
        # We want to cluster the *minority* class within X_real
        minority_real_samples = X_real_2d[y_real == self.minority_class_label]

        if len(minority_real_samples) == 0:
            # If no real minority samples, no cluster center can be formed. All synthetic samples are inconsistent.
            return np.zeros(len(X_syn_2d), dtype=bool)

        # KMeans needs at least n_clusters samples. We are using n_clusters=1.
        kmeans = KMeans(n_clusters=1, random_state=self.random_state, n_init=10)
        kmeans.fit(minority_real_samples)

        distances = np.linalg.norm(X_syn_2d - kmeans.cluster_centers_[0], axis=1)

        # Keep closer 70% of synthetic samples to the real minority cluster center
        # This threshold is applied to the distances of the synthetic samples themselves
        if len(distances) == 0:
            return np.array([])

        threshold = np.percentile(distances, 70)
        return distances <= threshold

    def _genetic_filter(self, X_real_2d, y_real, X_syn_2d, y_syn, test_size):
        # Ensure there are enough samples for stratified split in X_real
        unique_y_real = np.unique(y_real)
        if len(unique_y_real) < 2 or np.min(np.bincount(y_real.astype(int))) < 2:
            # Stratification not possible or not meaningful, fall back to simple split
            warnings.warn("X_real has less than 2 classes or not enough samples per class for stratified split. "
                          "Genetic filter will use non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(X_real_2d, y_real, test_size=test_size,
                                                              random_state=self.random_state)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_real_2d, y_real, test_size=test_size,
                                                              random_state=self.random_state, stratify=y_real)

        n_samples_syn = len(X_syn_2d)
        if n_samples_syn == 0:
            return np.array([])  # No synthetic samples to filter

        population_size = 10
        n_gen = 5
        pop = self.random_state.randint(0, 2, size=(population_size, n_samples_syn))

        best_fitness_overall = -1
        best_individual_overall = np.zeros(n_samples_syn, dtype=int)  # Default to no selection

        for gen_i in range(n_gen):
            fitness = []
            current_pop_valid_individuals = []  # Keep track of valid individuals in this generation
            for individual in pop:
                mask = individual == 1
                if mask.sum() == 0:  # Avoid empty synthetic set if it's the only added samples
                    fitness.append(0)
                    continue

                X_aug = np.concatenate([X_train, X_syn_2d[mask]])
                y_aug = np.concatenate([y_train, y_syn[mask]])

                # Check for sufficient classes in augmented training set for classification
                if len(np.unique(y_aug)) < 2:
                    fitness.append(0)
                    continue

                clf = RandomForestClassifier(random_state=self.random_state)
                clf.fit(X_aug, y_aug)

                # Check for sufficient classes in validation set and predictions for f1_score
                y_pred = clf.predict(X_val)
                if len(np.unique(y_val)) < 2 or len(np.unique(y_pred)) < 2:
                    fitness.append(0)
                    continue

                current_f1 = f1_score(y_val, y_pred, average='macro')
                fitness.append(current_f1)
                current_pop_valid_individuals.append(individual)  # Only add valid individuals

            if not fitness or max(fitness) == 0:
                warnings.warn(f"Genetic filter generation {gen_i} produced no valid individuals. Stopping early.")
                return best_individual_overall.astype(bool)  # Return the best found so far or all false

            # Update overall best
            current_gen_best_idx = np.argmax(fitness)
            if fitness[current_gen_best_idx] > best_fitness_overall:
                best_fitness_overall = fitness[current_gen_best_idx]
                best_individual_overall = pop[current_gen_best_idx]

            # Selection for next generation
            sorted_indices = np.argsort(fitness)
            # Ensure we select enough top individuals, handle cases where population_size//2 is 0 or 1
            num_to_select = max(1, population_size // 2)
            top_idx = sorted_indices[-num_to_select:]
            offspring = pop[top_idx]

            new_pop = []
            for p in offspring:
                mate = offspring[self.random_state.randint(len(offspring))]
                mask_crossover = self.random_state.rand(n_samples_syn) > 0.5
                child = np.where(mask_crossover, p, mate)
                if self.random_state.rand() < 0.2:  # Mutation probability
                    flip = self.random_state.randint(0, n_samples_syn)
                    child[flip] = 1 - child[flip]
                new_pop.append(child)

            # Fill remaining spots with random individuals to maintain population size if needed
            while len(new_pop) < population_size:
                new_pop.append(self.random_state.randint(0, 2, size=n_samples_syn))

            pop = np.array(new_pop)

        return best_individual_overall.astype(bool)

    def _de_filter(self, X_real_2d, y_real, X_syn_2d, y_syn, test_size):
        # Ensure there are enough samples for stratified split in X_real
        unique_y_real = np.unique(y_real)
        if len(unique_y_real) < 2 or np.min(np.bincount(y_real.astype(int))) < 2:
            warnings.warn("X_real has less than 2 classes or not enough samples per class for stratified split. "
                          "DE filter will use non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(X_real_2d, y_real, test_size=test_size,
                                                              random_state=self.random_state)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_real_2d, y_real, test_size=test_size,
                                                              random_state=self.random_state, stratify=y_real)

        n_samples_syn = len(X_syn_2d)
        if n_samples_syn == 0:
            return np.array([])  # No synthetic samples to filter

        pop_size = 10
        gens = 5
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        pop = self.random_state.rand(pop_size, n_samples_syn) * 2 - 1  # Initialize in range [-1, 1]

        best_fitness_overall = -1
        best_individual_overall_values = np.zeros(n_samples_syn)  # Default to no selection

        # Pre-calculate fitness for initial population
        pop_fitness = [0] * pop_size
        for i in range(pop_size):
            pop_fitness[i] = self._evaluate_de_individual(pop[i], X_train, y_train, X_syn_2d, y_syn, X_val, y_val)

        for gen_i in range(gens):
            new_pop = np.copy(pop)
            new_pop_fitness = list(pop_fitness)  # Copy fitness values too

            for i in range(pop_size):
                # Select three distinct random individuals (a, b, c) from the current population
                idxs = list(range(pop_size))
                idxs.pop(i)  # Ensure 'i' is not selected for a, b, c
                if len(idxs) < 3:  # Not enough individuals to pick 3 distinct ones, handle edge case
                    warnings.warn("Not enough individuals in DE population for distinct selection. Skipping mutation.")
                    continue

                a_idx, b_idx, c_idx = self.random_state.choice(idxs, 3, replace=False)
                a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]

                mutant = a + F * (b - c)

                cross_mask = self.random_state.rand(n_samples_syn) < CR
                # Ensure at least one element from mutant is picked
                if not np.any(cross_mask):
                    cross_mask[self.random_state.randint(n_samples_syn)] = True

                trial = np.where(cross_mask, mutant, pop[i])

                trial_fitness = self._evaluate_de_individual(trial, X_train, y_train, X_syn_2d, y_syn, X_val, y_val)

                if trial_fitness > new_pop_fitness[i]:
                    new_pop[i] = trial
                    new_pop_fitness[i] = trial_fitness

            pop = new_pop
            pop_fitness = new_pop_fitness  # Update fitness for next generation

            current_gen_best_idx = np.argmax(pop_fitness)
            if pop_fitness[current_gen_best_idx] > best_fitness_overall:
                best_fitness_overall = pop_fitness[current_gen_best_idx]
                best_individual_overall_values = pop[current_gen_best_idx]

        # If no good individual was found throughout the generations
        if best_fitness_overall == -1 and best_individual_overall_values.sum() == 0:
            warnings.warn("DE filter did not find any improving individuals. Returning an empty selection.")
            return np.zeros(n_samples_syn, dtype=bool)

        return (expit(best_individual_overall_values) > 0.5)

    def _evaluate_de_individual(self, ind_values, X_train, y_train, X_syn_2d, y_syn, X_val, y_val):
        """Helper function for DE to evaluate an individual's fitness."""
        mask = expit(ind_values) > 0.5
        if mask.sum() == 0:
            return 0
        X_aug = np.concatenate([X_train, X_syn_2d[mask]])
        y_aug = np.concatenate([y_train, y_syn[mask]])

        # Check for sufficient classes in augmented training set
        if len(np.unique(y_aug)) < 2:
            return 0

        clf = RandomForestClassifier(random_state=self.random_state)
        clf.fit(X_aug, y_aug)

        # Check for sufficient classes in validation set and predictions
        y_pred = clf.predict(X_val)
        if len(np.unique(y_val)) < 2 or len(np.unique(y_pred)) < 2:
            return 0
        return f1_score(y_val, y_pred, average='macro')