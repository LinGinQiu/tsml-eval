import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dense,
    Flatten,
    RepeatVector,
    Concatenate,
    Multiply,
    Lambda,
    Add,
)
from tensorflow.keras.utils import to_categorical
from aeon.transformations.collection import BaseCollectionTransformer
from sklearn.utils import check_random_state

class IBGANAugmenter(BaseCollectionTransformer):
    """
    IB-GAN style data augmenter for multivariate time series under class imbalance.

    Interface:
        - _fit(X, y): train G, D, C jointly on imbalanced data.
        - _transform(X, y): use trained generator to create synthetic series for minority
          classes and return augmented numpy arrays (original + synthetic).

    X shape: (n_samples, n_features/channel, sequence_length)
    y shape: (n_samples,), integer labels in [0, n_classes-1]
    """

    def __init__(
        self,
        epochs=500,
        batch_size=64,
        p_missing=0.1,
        p_hint=0.8,
        g_lr=1e-4,
        d_lr=1e-4,
        c_lr=1e-4,
        alpha=0.5,
        kernel_size=3,
        verbose=True,
        n_jobs=1,
        random_state=1,
    ):
        # These will be inferred in _fit
        self.n_classes = None
        self.sequence_length = None
        self.n_feature = None

        self.epochs = epochs
        self.batch_size = batch_size
        self.p_missing = p_missing
        self.p_hint = p_hint
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.c_lr = c_lr
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.verbose = verbose

        # Will be created in _build_models()
        self.generator = None
        self.discriminator = None
        self.classifier = None
        self.gan_model = None
        # separate optimizers for G, D, C
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.g_lr)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.d_lr)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.c_lr)
        self._fitted = False

        self.n_jobs = n_jobs
        self.random_state = random_state
        self._random_state = None

        self._generated_samples = None
        super().__init__()

    # ------------------------------------------------------------------
    # Public API for aeon/tsml-eval
    # ------------------------------------------------------------------
    def _fit(self, X, y=None):
        """
        Train IB-GAN on the given imbalanced dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features, sequence_length)
        y : np.ndarray, shape (n_samples,)
        """
        self._random_state = check_random_state(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if y is None:
            raise ValueError("IBGANAugmenter requires y for fitting.")

        # X is (n_samples, n_features/channel, sequence_length)
        n_samples, n_feature, sequence_length = X.shape
        self.sequence_length = sequence_length
        self.n_feature = n_feature

        # infer number of classes from y
        self.n_classes = int(np.unique(y).shape[0])

        if self.generator is None:
            self._build_models()

        y_onehot = to_categorical(y, num_classes=self.n_classes).astype(np.float32)

        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))
        if self.verbose:
            print(f"[IBGAN] n_samples={n_samples}, steps_per_epoch={steps_per_epoch}")

        for epoch in range(self.epochs):
            idx_all = self._random_state.permutation(n_samples)

            for step in range(steps_per_epoch):
                start = step * self.batch_size
                end = min((step + 1) * self.batch_size, n_samples)
                batch_idx = idx_all[start:end]

                x_real = X[batch_idx]  # (B, n_features, T)
                y_real_int = y[batch_idx]
                y_real = y_onehot[batch_idx]  # one-hot
                batch_size = x_real.shape[0]

                # (1) prepare inputs: (B, T, F)
                x_real_ts = np.transpose(x_real, (0, 2, 1))  # (B, T, F)

                M = self._sample_mask(
                    (batch_size, self.sequence_length, self.n_feature),
                    p_missing=self.p_missing,
                )
                hint = self._sample_hint_matrix(M, p_hint=self.p_hint)

                z_noise = self._random_state.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(batch_size, self.sequence_length, self.n_feature),
                ).astype(np.float32)

                x_masked = M * x_real_ts + (1.0 - M) * z_noise

                # (2) use current G to generate fake (numpy, used for D/C updates)
                fake_ts_tensor, _ = self.generator([x_masked, hint, M, y_real], training=False)
                fake_ts = fake_ts_tensor.numpy()  # (B, T, F)

                # ----------------------------------------------------
                # Step 1: train Discriminator (fix G, C)
                # ----------------------------------------------------
                with tf.GradientTape() as tape_d:
                    x_real_tf = tf.convert_to_tensor(x_real_ts)
                    fake_ts_tf = tf.convert_to_tensor(fake_ts)
                    hint_tf = tf.convert_to_tensor(hint)

                    d_real = self.discriminator(
                        [x_real_tf, hint_tf], training=True
                    )  # (B, 1)
                    d_fake = self.discriminator(
                        [fake_ts_tf, hint_tf], training=True
                    )  # (B, 1)

                    loss_real = tf.keras.losses.binary_crossentropy(
                        tf.ones_like(d_real), d_real
                    )
                    loss_fake = tf.keras.losses.binary_crossentropy(
                        tf.zeros_like(d_fake), d_fake
                    )
                    d_loss = tf.reduce_mean(loss_real + loss_fake)

                d_grads = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_variables)
                )

                # ----------------------------------------------------
                # Step 2: train Classifier (fix G, D)
                #   L_C = α * CE_real + (1-α) * E_x'[ w_D(x') * CE_fake(x') ]
                #   where w_D(x') = D(x') / (1 - D(x'))
                # ----------------------------------------------------
                with tf.GradientTape() as tape_c:
                    x_real_tf = tf.convert_to_tensor(x_real_ts)
                    fake_ts_tf = tf.convert_to_tensor(fake_ts)
                    hint_tf = tf.convert_to_tensor(hint)
                    y_real_tf = tf.convert_to_tensor(y_real)

                    c_real = self.classifier(
                        [x_real_tf, hint_tf], training=True
                    )  # (B, n_classes)
                    c_fake = self.classifier(
                        [fake_ts_tf, hint_tf], training=True
                    )  # (B, n_classes)

                    # per-sample CE
                    ce_real = tf.keras.losses.categorical_crossentropy(
                        y_real_tf, c_real
                    )  # (B,)
                    ce_fake = tf.keras.losses.categorical_crossentropy(
                        y_real_tf, c_fake
                    )  # (B,)

                    # compute w_D for fake: w_D = D(fake) / (1 - D(fake))
                    d_fake_for_w = self.discriminator(
                        [fake_ts_tf, hint_tf], training=False
                    )  # (B,1)
                    eps = 1e-8
                    w_D = d_fake_for_w / (1.0 - d_fake_for_w + eps)  # (B,1)
                    w_D = tf.squeeze(w_D, axis=-1)  # (B,)

                    weighted_fake_ce = w_D * ce_fake

                    ce_real_mean = tf.reduce_mean(ce_real)
                    ce_fake_mean = tf.reduce_mean(weighted_fake_ce)

                    c_loss = self.alpha * ce_real_mean + (1.0 - self.alpha) * ce_fake_mean

                c_grads = tape_c.gradient(c_loss, self.classifier.trainable_variables)
                self.c_optimizer.apply_gradients(
                    zip(c_grads, self.classifier.trainable_variables)
                )

                # ----------------------------------------------------
                # Step 3: train Generator (fix D, C parameters; update G)
                #   L_G = λ_adv * L_adv + λ_cls * (1-α) * E_x'[ w_D(x') * CE_fake(x') ]
                #   - adversarial: encourage D(G(...)) -> 1
                #   - classification: encourage C(G(...)) to predict y
                # ----------------------------------------------------
                with tf.GradientTape() as tape_g:
                    x_masked_tf = tf.convert_to_tensor(x_masked)
                    hint_tf = tf.convert_to_tensor(hint)
                    M_tf = tf.convert_to_tensor(M)
                    y_real_tf = tf.convert_to_tensor(y_real)

                    fake_ts_g, _ = self.generator(
                        [x_masked_tf, hint_tf, M_tf, y_real_tf], training=True
                    )  # (B, T, F)

                    d_fake_g = self.discriminator(
                        [fake_ts_g, hint_tf], training=False
                    )  # (B,1)
                    c_fake_g = self.classifier(
                        [fake_ts_g, hint_tf], training=False
                    )  # (B,n_classes)

                    # adversarial term: D(fake) -> 1
                    adv_loss = tf.keras.losses.binary_crossentropy(
                        tf.ones_like(d_fake_g), d_fake_g
                    )
                    adv_loss = tf.reduce_mean(adv_loss)

                    # classification term with w_D
                    ce_fake_g = tf.keras.losses.categorical_crossentropy(
                        y_real_tf, c_fake_g
                    )  # (B,)
                    w_D_g = d_fake_g / (1.0 - d_fake_g + 1e-8)  # (B,1)
                    w_D_g = tf.squeeze(w_D_g, axis=-1)  # (B,)
                    cls_loss = tf.reduce_mean(w_D_g * ce_fake_g)

                    lambda_adv = 1.0
                    lambda_cls = 1.0
                    g_total_loss = lambda_adv * adv_loss + lambda_cls * (1.0 - self.alpha) * cls_loss

                g_grads = tape_g.gradient(g_total_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(
                    zip(g_grads, self.generator.trainable_variables)
                )

                if self.verbose and step % 50 == 0:
                    print(
                        f"[Epoch {epoch + 1}/{self.epochs} Step {step + 1}/{steps_per_epoch}] "
                        f"D_loss={float(d_loss):.4f} "
                        f"C_loss={float(c_loss):.4f} "
                        f"G_loss={float(g_total_loss.numpy()):.4f} "
                        f"(adv={float(adv_loss.numpy()):.4f}, cls={float(cls_loss.numpy()):.4f})"
                    )

        self._fitted = True
        if self.verbose:
            print("[IBGAN] Training finished.")
        return self

    def _transform(self, X, y=None):
        """
        Generate synthetic samples for minority classes to balance the dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features, sequence_length)
        y : np.ndarray, shape (n_samples,)
            If None, just returns X (no balancing).

        Returns
        -------
        X_aug : np.ndarray
        y_aug : np.ndarray
            Original + synthetic samples, both as numpy arrays (no tensors).
        """
        if not self._fitted:
            raise RuntimeError("You must call fit() before transform().")

        X = np.asarray(X, dtype=np.float32)
        if y is None:
            return X

        y = np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]

        # Count per class
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_synth_list = []
        y_synth_list = []

        for cls, cnt in zip(classes, counts):
            if cnt >= max_count:
                continue
            # Number of synthetic samples needed
            n_needed = max_count - cnt

            # Sample existing indices (with replacement) for this class
            idx_cls = np.where(y == cls)[0]
            chosen_idx = self._random_state.choice(idx_cls, size=n_needed, replace=True)

            x_real = X[chosen_idx]  # (B, n_features, T)
            y_real_int = y[chosen_idx]
            y_real = to_categorical(
                y_real_int, num_classes=self.n_classes
            ).astype(np.float32)

            batch_size = x_real.shape[0]

            # For Conv1D, transpose to (B, T, F)
            x_real_ts = np.transpose(x_real, (0, 2, 1))

            # Build masks and hints
            M = self._sample_mask(
                (batch_size, self.sequence_length, self.n_feature),
                p_missing=self.p_missing,
            )
            hint = self._sample_hint_matrix(M, p_hint=self.p_hint)

            z_noise = self._random_state.normal(
                loc=0.0,
                scale=1.0,
                size=(batch_size, self.sequence_length, self.n_feature),
            ).astype(np.float32)

            x_masked = M * x_real_ts + (1.0 - M) * z_noise

            fake_ts, _ = self.generator.predict(
                [x_masked, hint, M, y_real], verbose=0
            )  # (B, T, F)

            # 转回到 (B, n_features, T)
            fake_ts_ch_first = np.transpose(fake_ts, (0, 2, 1))

            X_synth_list.append(fake_ts_ch_first)
            y_synth_list.append(y_real_int)

        if len(X_synth_list) == 0:
            # Already balanced
            return X, y

        X_synth = np.concatenate(X_synth_list, axis=0)
        y_synth = np.concatenate(y_synth_list, axis=0)
        self._generated_samples = X_synth
        X_aug = np.concatenate([X, X_synth], axis=0)
        y_aug = np.concatenate([y, y_synth], axis=0)
        import gc
        import tensorflow as tf

        # 删除模型对象引用
        del self.generator
        del self.discriminator
        del self.classifier
        self.generator = None
        self.discriminator = None
        self.classifier = None

        # 清除 Keras 后端状态
        tf.keras.backend.clear_session()

        # 强制 Python 垃圾回收
        gc.collect()
        # Shuffle after augmentation
        idx = self._random_state.permutation(X_aug.shape[0])
        X_aug = X_aug[idx]
        y_aug = y_aug[idx]

        return X_aug.astype(np.float32), y_aug.astype(np.int64)

    # ------------------------------------------------------------------
    # Internal: model building
    # ------------------------------------------------------------------
    def _build_models(self):
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.classifier = self._build_classifier()

        # D, C 各自有独立 optimizer，用于自己的 train_on_batch
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.d_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.c_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 不再在这里动 trainable 标志（保持 True）
        # 不再依赖 gan_model 来训练 G，而是自己写 GradientTape

        if self.verbose:
            print("[IBGAN] Generator summary:")
            self.generator.summary()
            print("[IBGAN] Discriminator summary:")
            self.discriminator.summary()
            print("[IBGAN] Classifier summary:")
            self.classifier.summary()

    def _build_generator(self):
        # Inputs
        missing_shape = (self.sequence_length, self.n_feature)
        mask_shape = (self.sequence_length, self.n_feature)
        labels_shape = (self.n_classes,)
        hint_shape = (self.sequence_length, self.n_feature)

        in_ts = Input(shape=missing_shape, name="g_ts_in")
        in_hint = Input(shape=hint_shape, name="g_hint_in")
        in_labels = Input(shape=labels_shape, name="g_labels_in")
        in_mask = Input(shape=mask_shape, name="g_mask_in")

        # Tile labels over time dimension
        labels_tile = RepeatVector(self.sequence_length)(in_labels)

        merged = Concatenate(axis=2)([in_ts, in_hint, labels_tile])

        filters1 = self.n_feature * 2

        x = Conv1D(
            filters=filters1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )(merged)
        x = Conv1D(
            filters=self.n_feature,
            kernel_size=self.kernel_size,
            padding="same",
            activation=None,
        )(x)

        fake_ts = x

        masked_data = Multiply()([in_ts, in_mask])
        rev_mask = Lambda(lambda t: 1.0 - t)(in_mask)
        imputed_vals = Multiply()([fake_ts, rev_mask])
        imputed_ts = Add(name="imputed_ts")([masked_data, imputed_vals])

        model = Model(
            inputs=[in_ts, in_hint, in_mask, in_labels],
            outputs=[imputed_ts, in_hint],
            name="Generator",
        )
        return model

    def _build_discriminator(self):
        ts_shape = (self.sequence_length, self.n_feature)
        hint_shape = (self.sequence_length, self.n_feature)

        in_ts = Input(shape=ts_shape, name="d_ts_in")
        in_hint = Input(shape=hint_shape, name="d_hint_in")

        merged = Concatenate(axis=2)([in_ts, in_hint])

        filters1 = self.n_feature * 2

        x = Conv1D(
            filters=filters1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )(merged)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(
            filters=self.n_feature,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        out = Dense(1, activation="sigmoid", name="d_out")(x)

        model = Model(inputs=[in_ts, in_hint], outputs=out, name="Discriminator")
        return model

    def _build_classifier(self):
        ts_shape = (self.sequence_length, self.n_feature)
        hint_shape = (self.sequence_length, self.n_feature)

        in_ts = Input(shape=ts_shape, name="c_ts_in")
        in_hint = Input(shape=hint_shape, name="c_hint_in")

        # In the official example, hint is usually concatenated; here we keep it simple
        merged = Concatenate(axis=2)([in_ts, in_hint])

        filters1 = self.n_feature * 2

        x = Conv1D(
            filters=filters1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )(merged)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        x = Flatten()(x)
        x = Dense(self.n_classes * 2, activation="relu")(x)
        out = Dense(self.n_classes, activation="softmax", name="c_out")(x)

        model = Model(inputs=[in_ts, in_hint], outputs=out, name="Classifier")
        return model

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _sample_mask(self, shape, p_missing):
        """Binary mask matrix M ~ Bernoulli(1 - p_missing)."""
        mask = self._random_state.rand(*shape) > p_missing
        return mask.astype(np.float32)

    def _sample_hint_matrix(self, mask_matrix, p_hint=0.8):
        """Hint matrix as in GAIN/IB-GAN: subset of mask entries revealed to D."""
        hints = self._random_state.rand(*mask_matrix.shape) > p_hint
        hints = hints.astype(np.float32)
        hint_matrix = hints * mask_matrix
        return hint_matrix


if __name__ == "__main__":
    dataset_name = 'AllGestureWiimoteX_eq'
    smote = IBGANAugmenter()
    # Example usage
    from local.load_ts_data import load_ts_data

    X_train, y_train, X_test, y_test = load_ts_data(dataset_name)
    print(np.unique(y_train, return_counts=True))
    arr = X_test
    # 检查是否有 NaN
    print(np.isnan(arr).any())  # True

    X_resampled, y_resampled = smote.fit_transform(X_train, y_train)
    print(X_resampled.shape)
    print(np.unique(y_resampled, return_counts=True))
    # # Example usage
    # dataset_name = 'yanran'
    # from local.baseline_rf_classifier import load_samples_from_directory, extract_raw_features
    # from pathlib import Path
    #
    # dataset_path = Path('/Users/qiuchuanhang/PycharmProjects/tsml-eval/local')
    # train_dangerous_dir = dataset_path / "train" / "dangerous"
    # train_normal_dir = dataset_path / "train" / "normal"
    # train_dangerous_orig = load_samples_from_directory(train_dangerous_dir)
    # train_normal = load_samples_from_directory(train_normal_dir)
    # train_dangerous = np.array([extract_raw_features(s) for s in train_dangerous_orig])
    # train_normal = np.array([extract_raw_features(s) for s in train_normal])
    # X_tr = np.concatenate([train_dangerous, train_normal], axis=0)
    # y_tr = np.array([1] * len(train_dangerous_orig) + [0] * len(train_normal))
    # X_tr = X_tr[:, np.newaxis, :]  # (N, C=1, T)
    # try:
    #     X_resampled, y_resampled = np.load('/Users/qiuchuanhang/PycharmProjects/tsml-eval/local/yanran')
    # except FileNotFoundError:
    #     smote = IBGANAugmenter(epochs=10, )
    #     X_resampled, y_resampled = smote.fit_transform(X_tr, y_tr)
    #     X_resampled = X_resampled.squeeze()
    #     print(f'Original dataset shape {np.unique(y_tr, return_counts=True)}')
    #     print(f'Resampled dataset shape {np.unique(y_resampled, return_counts=True)}')
    #     print(f'Resampled X shape {X_resampled.shape,}')
    # eval_dangerous_dir = dataset_path / "eval" / "dangerous"
    # eval_normal_dir = dataset_path / "eval" / "normal"
    #
    # eval_dangerous, eval_dangerous_tof_paths, eval_dangerous_rs_paths = load_samples_from_directory(eval_dangerous_dir,
    #                                                                                                 return_paths=True)
    # eval_normal, eval_normal_tof_paths, eval_normal_rs_paths = load_samples_from_directory(eval_normal_dir,
    #                                                                                        return_paths=True)
    #
    # X_eval_dangerous = np.array([extract_raw_features(s) for s in eval_dangerous])
    # X_eval_normal = np.array([extract_raw_features(s) for s in eval_normal])
    # X_eval = np.vstack([X_eval_dangerous, X_eval_normal])
    # y_eval = np.array([1] * len(eval_dangerous) + [0] * len(eval_normal))
    #
    # from local.baseline_rf_classifier import train_rf_classifier, evaluate_classifier
    #
    # clf = train_rf_classifier(
    #     X_tr.squeeze(), y_tr,
    #     n_estimators=100,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     random_state=42
    # )
    # print("\n" + "=" * 60)
    # print("EVALUATION RESULTS of ORIGINAL DATA")
    # print("=" * 60)
    # threshold = 0.4
    #
    # results = evaluate_classifier(clf, X_eval, y_eval, threshold=threshold)
    #
    # print(f"\nClassification threshold: {threshold}")
    # print(f"Confusion Matrix:")
    # print(results['confusion_matrix'])
    # clf = train_rf_classifier(
    #     X_resampled, y_resampled,
    #     n_estimators=100,
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     random_state=42
    # )
    # print("EVALUATION RESULTS of RESAMPLED DATA")
    # results = evaluate_classifier(clf, X_eval, y_eval, threshold=threshold)
    #
    # print(results['confusion_matrix'])
