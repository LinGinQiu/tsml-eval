import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist, pdist, squareform
from matplotlib.path import Path
from collections import OrderedDict
from typing import Union, Optional
from aeon.transformations.collection import BaseCollectionTransformer


class HS_SMOTE(BaseCollectionTransformer):
    """
    Over-sampling using the Hexagonal-Structure SMOTE (HS-SMOTE).
    Converted from MATLAB implementation.

    Input X : ndarray of shape (n_samples, n_channels, seq_len) or (n_samples, seq_len)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "requires_y": True,
    }

    def __init__(
            self,
            n_neighbors=5,  # 保留参数，虽然HS-SMOTE主要依赖网格邻居
            random_state=None,
    ):
        self.random_state = random_state
        self.n_neighbors = n_neighbors

        self._random_state = None
        self.sampling_strategy_ = None
        super().__init__()

    def _fit(self, X, y=None):
        """
        Logic to determine how many samples to generate per class.
        """
        self._random_state = check_random_state(self.random_state)

        # Validate y
        if y is None:
            raise ValueError("y is required for HS-SMOTE")

        # generate sampling target by targeting all classes except the majority
        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)

        # Dictionary: {class_label: n_samples_to_generate}
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }
        self.sampling_strategy_ = OrderedDict(sorted(sampling_strategy.items()))

        return self

    def _transform(self, X, y=None):
        """
        Performs the HS-SMOTE generation logic.
        """
        if y is None:
            raise ValueError("y is required for transform in HS-SMOTE")

        # 1. Handle Input Shapes (3D Time Series -> 2D Tabular)
        X_in = np.asarray(X)
        original_shape = X_in.shape
        if X_in.ndim == 3:
            n_samples, n_channels, seq_len = X_in.shape
            X_2d = X_in.reshape(n_samples, -1)
        else:
            n_samples, n_features = X_in.shape
            X_2d = X_in
            n_channels = 1
            seq_len = n_features

        # Prepare list to collect new data
        X_new_list = [X_2d]
        y_new_list = [y]

        # Loop through each minority class defined in _fit
        for class_label, n_samples_needed in self.sampling_strategy_.items():
            if n_samples_needed <= 0:
                continue
            print(f"Oversampling class {class_label}: generating {n_samples_needed} samples using HS_SMOTE...")
            # --- HS-SMOTE Logic Starts Here ---

            # Identify indices
            min_indices = np.where(y == class_label)[0]
            maj_indices = np.where(y != class_label)[0]

            # Data for PCA (normalize/z-score as in matlab 'zscore')
            # Note: We compute zscore on the whole dataset for PCA mapping
            X_mean = np.mean(X_2d, axis=0)
            X_std = np.std(X_2d, axis=0)
            X_std[X_std == 0] = 1  # avoid division by zero
            X_norm = (X_2d - X_mean) / X_std

            # PCA Projection to 2D
            pca = PCA(n_components=2, random_state=self._random_state)
            score = pca.fit_transform(X_norm)
            res = np.column_stack((score, y))  # col 0: x, col 1: y, col 2: label

            x_pca = res[:, 0]
            y_pca = res[:, 1]

            # Calculate rc (Grid Size) based on minority class density
            # MATLAB: mean(min_a(:))/mean(a(:)) logic simplified for Python
            # Calculate pairwise distances for all points
            dist_matrix = pdist(res[:, :2])
            mean_dist_all = np.mean(dist_matrix)

            # Calculate pairwise distances for minority points
            min_points_pca = res[min_indices, :2]
            if len(min_indices) > 1:
                min_dist_matrix = pdist(min_points_pca)
                mean_dist_min = np.mean(min_dist_matrix)
            else:
                mean_dist_min = 0  # Handle single point case

            if mean_dist_all == 0: mean_dist_all = 1e-6

            # Factors from MATLAB code
            ccsb = mean_dist_min / mean_dist_all if mean_dist_all > 0 else 1

            # Nearest neighbor distance logic (ccsa)
            # MATLAB: [minv ind]=min(a,[],2);
            nn_all = NearestNeighbors(n_neighbors=2).fit(res[:, :2])
            dists_all, _ = nn_all.kneighbors(res[:, :2])
            mean_nn_all = np.mean(dists_all[:, 1])  # 2nd column is nearest neighbor (not self)

            nn_min = NearestNeighbors(n_neighbors=2).fit(min_points_pca)
            if len(min_indices) > 1:
                dists_min, _ = nn_min.kneighbors(min_points_pca)
                mean_nn_min = np.mean(dists_min[:, 1])
            else:
                mean_nn_min = 0

            # p1 calculation (from code: p1 = ccsb)
            p1 = ccsb
            rc = mean_nn_all * p1 if mean_nn_all > 0 else 0.5  # Fallback
            if rc == 0: rc = 1.0

            # Hexagon Parameters
            dy = 2 * rc
            dx = rc * np.sqrt(3)

            # Grid Bounds
            min_y_pca, max_y_pca = np.min(y_pca), np.max(y_pca)
            min_x_pca, max_x_pca = np.min(x_pca), np.max(x_pca)

            # Generate Grid and Domains
            # MATLAB loops yk then xk
            domains = []  # List of dicts: {center, polygon, point_indices, type}

            # Create hexagonal grid points
            # yfun logic: yp = sqrt(3)*x/3 + yk. This implies a staggered grid.
            # Simplified generator:
            y_range = np.arange(-max_y_pca - dy, max_y_pca + dy, dy)
            x_range = np.arange(-max_x_pca - dx, max_x_pca + dx, dx)

            domain_id_counter = 0

            for yk in y_range:
                # Based on MATLAB: yp = sqrt(3)*xk/3 + yk is effectively shifting rows
                # Standard Hex grid logic: odd rows shifted by dx/2
                for xk in x_range:
                    # Construct Hexagon vertices
                    # MATLAB: T=[xp+1i*yp]+rc*exp(1i*AI)*2/sqrt(3);
                    # AI=pi/3*[1:7]

                    # Center (approximate based on loop logic)
                    xp = xk
                    yp = (np.sqrt(3) * xp / 3) + yk

                    # Check if center is roughly in data bounds
                    if (-max_x_pca - dx < xp < max_x_pca + dx and
                            -max_y_pca - dy < yp < max_y_pca + dy):

                        # Generate vertices
                        angles = np.linspace(np.pi / 3, 7 * np.pi / 3, 7)[:-1]  # 6 points
                        # MATLAB factor: 2/sqrt(3) makes the hexagon fit the circle better
                        r_hex = rc * 2 / np.sqrt(3)

                        hex_x = xp + r_hex * np.cos(angles)
                        hex_y = yp + r_hex * np.sin(angles)
                        polygon_verts = np.column_stack((hex_x, hex_y))

                        # Check points inside
                        path = Path(polygon_verts)
                        points_inside_mask = path.contains_points(res[:, :2])
                        indices_inside = np.where(points_inside_mask)[0]

                        if len(indices_inside) > 0:
                            # Determine domain type
                            labels_inside = y[indices_inside]
                            has_min = np.any(labels_inside == class_label)
                            has_maj = np.any(labels_inside != class_label)

                            dtype = 'empty'
                            if has_min and not has_maj:
                                dtype = 'minority'  # YU1
                            elif not has_min and has_maj:
                                dtype = 'majority'  # YU2
                            else:
                                dtype = 'disputed'  # YUdis

                            domains.append({
                                'id': domain_id_counter,
                                'center': [xp, yp],
                                'indices': indices_inside,
                                'labels': labels_inside,
                                'type': dtype,
                                'poly': polygon_verts
                            })
                            domain_id_counter += 1

            if not domains:
                continue

            # Domain Processing (Neighbors and Retyping)
            n_domains = len(domains)
            domain_centers = np.array([d['center'] for d in domains])
            dist_dom = squareform(pdist(domain_centers))

            # Helper to get domain indices by type
            def get_ids_by_type(t):
                return [i for i, d in enumerate(domains) if d['type'] == t]

            YU1 = get_ids_by_type('minority')
            YU2 = get_ids_by_type('majority')
            YUdis = get_ids_by_type('disputed')
            YU0 = []  # Empty domains not stored in list, simplified logic

            # Re-classify domains based on neighbors (Logic from loop NUM2)
            # Find 6 nearest neighbors for each domain
            # We use a copy of types to update
            current_types = [d['type'] for d in domains]

            for i in range(n_domains):
                # Sort distances
                dists = dist_dom[i]
                neighbor_indices = np.argsort(dists)[1:7]  # Skip self (index 0)

                # Count neighbor types
                nb_types = [current_types[n] for n in neighbor_indices if n < len(current_types)]
                n_min = nb_types.count('minority')
                n_maj = nb_types.count('majority')

                if current_types[i] == 'majority' and n_min == 6:
                    current_types[i] = 'minority'  # Flip to 1
                elif current_types[i] == 'minority' and n_maj == 6:
                    current_types[i] = 'majority'  # Flip to -1

            # Update types in list
            for i, d in enumerate(domains): d['type'] = current_types[i]

            # Scoring / Probability Calculation (Logic from loop NUM3)
            P = np.zeros(n_domains)
            for i in range(n_domains):
                dists = dist_dom[i]
                neighbor_indices = np.argsort(dists)[1:7]

                score = 0
                base_score = 0

                if domains[i]['type'] == 'minority':
                    base_score = 1
                    for nb in neighbor_indices:
                        if nb < n_domains:
                            if domains[nb]['type'] == 'minority':
                                score += 0.25
                            elif domains[nb]['type'] == 'disputed':
                                score += 0.1
                elif domains[i]['type'] == 'disputed':
                    base_score = 0.5
                    for nb in neighbor_indices:
                        if nb < n_domains:
                            if domains[nb]['type'] == 'minority':
                                score += 0.25
                            elif domains[nb]['type'] == 'disputed':
                                score += 0.1

                P[i] = base_score + score

            # Final Probability (P2 calculation)
            p2_factor = 1.0  # From code p2=1
            P2 = np.zeros(n_domains)
            for i in range(n_domains):
                # If domain has no points or is majority, prob is 0
                if len(domains[i]['indices']) == 0 or domains[i]['type'] == 'majority':
                    P2[i] = 0
                else:
                    P2[i] = P[i] * (p2_factor ** P[i])

            if np.sum(P2) == 0:
                # Fallback if no valid domains found (rare)
                prob = np.ones(n_domains) / n_domains
            else:
                prob = P2 / np.sum(P2)

            # Sample Generation
            generated_samples = []
            generated_labels = []

            p3 = 0  # point quantity weight factor

            count_generated = 0
            while count_generated < n_samples_needed:
                # 1. Select Domain S1
                s1_idx = self._random_state.choice(n_domains, p=prob)

                # 2. Select point K1 from S1 (only minority points)
                s1_indices = domains[s1_idx]['indices']
                s1_labels = y[s1_indices]
                valid_k1 = s1_indices[s1_labels == class_label]

                if len(valid_k1) == 0: continue
                k1_idx = self._random_state.choice(valid_k1)

                # 3. Select Neighbor Domain S2
                dists = dist_dom[s1_idx]
                neighbor_indices = np.argsort(dists)[1:7]

                # Filter neighbors that are valid (not majority/empty)
                valid_neighbors = []
                for nb in neighbor_indices:
                    if nb < n_domains and domains[nb]['type'] != 'majority' and len(domains[nb]['indices']) > 0:
                        valid_neighbors.append(nb)

                if not valid_neighbors: continue

                # Select one neighbor randomly (uniform in code)
                # Code: K2(NUM7) logic implies picking one valid neighbor
                s2_idx = self._random_state.choice(valid_neighbors)

                # 4. Select point K2 from S2
                s2_indices = domains[s2_idx]['indices']
                s2_labels = y[s2_indices]
                valid_k2 = s2_indices[s2_labels == class_label]

                if len(valid_k2) == 0: continue
                k2_idx = self._random_state.choice(valid_k2)

                # 5. Interpolate
                # MATLAB: Enew(NUM8+1,:)=Enew(NUM8,:)+rand(1)*(p5/10)*(ecoli(K(NUM8+1),:)-(Enew(NUM8,:)));
                # Simplified: New = K1 + rand * (K2 - K1)
                # Note: MATLAB code has a weird loop `NUM8` but effectively interpolates between points.
                # Since we select just pairs here, we do standard linear interpolation.

                # Heuristic for interpolation magnitude (p5/10)
                # Code: p5=floor(10*(p4^p3)), p4=1 -> p5=10 -> factor = 1.0 * rand
                rand_val = self._random_state.rand()

                # Interpolate in Original Feature Space (X_2d)
                sample_k1 = X_2d[k1_idx]
                sample_k2 = X_2d[k2_idx]

                new_sample = sample_k1 + rand_val * (sample_k2 - sample_k1)

                # 6. Filter: Check if new sample is valid
                # Map new sample to PCA space to check domain
                # We need to project new_sample using the fitted PCA
                new_sample_norm = (new_sample - X_mean) / X_std
                new_sample_pca = pca.transform(new_sample_norm.reshape(1, -1))[0]

                # Check which domain it falls into
                # Only keep if surrounding domains have minority points
                valid_generation = False

                # Find the domain this point falls into (approximate by nearest center to avoid heavy geometry)
                dists_to_centers = np.linalg.norm(domain_centers - new_sample_pca, axis=1)
                nearest_dom_idx = np.argmin(dists_to_centers)

                # Check nearest domain logic
                # MATLAB: checks if surrounding 6 domains of the new point location have minority data
                nb_indices = np.argsort(dist_dom[nearest_dom_idx])[0:7]  # include self

                # Check if any neighbor domain is minority or disputed
                for nb in nb_indices:
                    if nb < n_domains:
                        if domains[nb]['type'] in ['minority', 'disputed']:
                            valid_generation = True
                            break

                if valid_generation:
                    generated_samples.append(new_sample)
                    generated_labels.append(class_label)
                    count_generated += 1

            if generated_samples:
                X_new_list.append(np.array(generated_samples))
                y_new_list.append(np.array(generated_labels))

        # Reassemble data
        X_resampled_2d = np.vstack(X_new_list)
        y_resampled = np.hstack(y_new_list)

        # Reshape back to original 3D format if needed
        if original_shape is not None and len(original_shape) == 3:
            X_resampled = X_resampled_2d.reshape(X_resampled_2d.shape[0], n_channels, seq_len)
        else:
            X_resampled = X_resampled_2d

        return X_resampled, y_resampled

if __name__ == "__main__":
    global leng
    dataset_name = 'Covid3Month_disc'
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    print(np.unique(y_train, return_counts=True))
    # _plot_series_list([X_majority[0][0][:leng], X_majority[1][0][:leng]], title="Majority class example")
    # _plot_series_list([X_majority[2][0][:leng], X_majority[3][0][:leng]], title="Majority class example")
    smote = HS_SMOTE(
        random_state=42,
            )

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    stop = ""
