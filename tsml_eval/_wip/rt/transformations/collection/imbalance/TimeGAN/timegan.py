"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py (Upgraded to TensorFlow 2.x)

Note: Use original data as training set to generater synthetic data (time-series)
"""

import tensorflow as tf
import numpy as np
from tsml_eval._wip.rt.transformations.collection.imbalance.TimeGAN.utils import extract_time, random_generator, batch_generator
import os
import socket
hostname = socket.gethostname()
is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名
if is_mac:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def timegan(ori_data, parameters):
    """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
    # 1. Data processing
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val

    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # 确保数据是 float32 类型，这是 TF2 的首选类型
    ori_data = ori_data.astype(np.float32)

    # 2. Build Networks (Native TF2 Keras)
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # 选择 RNN 单元类型
    if module_name == 'gru':
        RNN_Layer = tf.keras.layers.GRU
    elif module_name == 'lstm':
        RNN_Layer = tf.keras.layers.LSTM
    elif module_name == 'lstmLN':
        print("Warning: LayerNorm LSTM not natively supported in simple Keras port. Falling back to standard LSTM.")
        RNN_Layer = tf.keras.layers.LSTM
    else:
        RNN_Layer = tf.keras.layers.GRU

    def make_rnn_model(input_dim, output_dim, num_layers, activation='tanh', output_activation=None, name='model'):
        model = tf.keras.Sequential(name=name)
        model.add(tf.keras.layers.InputLayer(input_shape=(max_seq_len, input_dim)))
        for i in range(num_layers):
            # return_sequences=True 对应 TF1 dynamic_rnn 的行为
            model.add(RNN_Layer(units=hidden_dim, return_sequences=True, activation=activation))

        model.add(tf.keras.layers.Dense(units=output_dim, activation=output_activation))
        return model

    # Define the 4 main components
    # 1. Embedder: Input -> Latent (H)
    embedder = make_rnn_model(dim, hidden_dim, num_layers, output_activation='sigmoid', name='embedder')

    # 2. Recovery: Latent (H) -> Output (X_tilde)
    recovery = make_rnn_model(hidden_dim, dim, num_layers, output_activation='sigmoid', name='recovery')

    # 3. Generator: Random (Z) -> Latent (E)
    generator = make_rnn_model(dim, hidden_dim, num_layers, output_activation='sigmoid', name='generator')

    # 4. Supervisor: Latent (H) -> Latent (S) (Next step prediction)
    # 注意：supervisor 通常比其他网络少一层，原代码是 num_layers-1
    supervisor = make_rnn_model(hidden_dim, hidden_dim, num_layers - 1, output_activation='sigmoid', name='supervisor')

    # 5. Discriminator: Latent (H) -> Real/Fake (Y)
    discriminator = make_rnn_model(hidden_dim, 1, num_layers, output_activation=None, name='discriminator')

    # 3. Optimizers
    # 使用 TF2 的 Adam 优化器
    embedder0_optimizer = tf.keras.optimizers.Adam()
    embedder_optimizer = tf.keras.optimizers.Adam()
    gen_optimizer = tf.keras.optimizers.Adam()
    disc_optimizer = tf.keras.optimizers.Adam()
    gs_optimizer = tf.keras.optimizers.Adam() # 专门用于 Supervised loss 的优化器

    # 4. Loss Functions
    mse = tf.keras.losses.MeanSquaredError()
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 5. Training Steps (使用 @tf.function 加速)

    # @tf.function
    def train_embedder(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            x_tilde = recovery(h)
            loss_e0 = 10 * tf.sqrt(mse(x, x_tilde))

        # 联合训练 embedder 和 recovery
        vars_e = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(loss_e0, vars_e)
        embedder0_optimizer.apply_gradients(zip(grads, vars_e))
        return loss_e0

    # @tf.function
    def train_supervisor(x, z):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervise = supervisor(h)
            # Supervised Loss
            loss_s = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])

        # 【修改这里】去掉 generator.trainable_variables
        # 在 Supervised Training 阶段，我们只训练 Supervisor
        vars_s = supervisor.trainable_variables
        grads = tape.gradient(loss_s, vars_s)
        gs_optimizer.apply_gradients(zip(grads, vars_s))
        return loss_s

    # @tf.function
    def train_generator(x, z):
        with tf.GradientTape() as tape:
            # 1. Generator Forward
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            h_hat_supervise = supervisor(embedder(x)) # 用于计算 supervised loss

            # 2. Discriminator Forward (Fake only)
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            # 3. Losses
            # Adversarial Loss (Unsupervised)
            loss_u = bce_logits(tf.ones_like(y_fake), y_fake)
            loss_u_e = bce_logits(tf.ones_like(y_fake_e), y_fake_e)

            # Supervised Loss
            loss_s = mse(embedder(x)[:, 1:, :], h_hat_supervise[:, :-1, :])

            # Moments Loss (Mean & Std)
            x_hat = recovery(h_hat)
            x_mean, x_var = tf.nn.moments(x, axes=[0])
            x_hat_mean, x_hat_var = tf.nn.moments(x_hat, axes=[0])
            loss_v1 = tf.reduce_mean(tf.abs(tf.sqrt(x_hat_var + 1e-6) - tf.sqrt(x_var + 1e-6)))
            loss_v2 = tf.reduce_mean(tf.abs(x_hat_mean - x_mean))
            loss_v = loss_v1 + loss_v2

            # Total Generator Loss
            g_loss = loss_u + gamma * loss_u_e + 100 * tf.sqrt(loss_s) + 100 * loss_v

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(g_loss, vars_g)
        gen_optimizer.apply_gradients(zip(grads, vars_g))
        return g_loss, loss_u, loss_s, loss_v

    # @tf.function
    def train_embedder_joint(x, z):
        # 在联合训练阶段，Embedding 网络的 loss 包含重构 loss 和 supervised loss
        with tf.GradientTape() as tape:
            h = embedder(x)
            x_tilde = recovery(h)
            h_hat_supervise = supervisor(h)

            loss_e0 = 10 * tf.sqrt(mse(x, x_tilde))
            loss_s = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            e_loss = loss_e0 + 0.1 * loss_s

        vars_er = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(e_loss, vars_er)
        embedder_optimizer.apply_gradients(zip(grads, vars_er))
        return loss_e0

    # @tf.function
    def train_discriminator(x, z):
        with tf.GradientTape() as tape:
            # Real
            h = embedder(x)
            y_real = discriminator(h)

            # Fake
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            # Discriminator Losses
            loss_d_real = bce_logits(tf.ones_like(y_real), y_real)
            loss_d_fake = bce_logits(tf.zeros_like(y_fake), y_fake)
            loss_d_fake_e = bce_logits(tf.zeros_like(y_fake_e), y_fake_e)

            d_loss = loss_d_real + loss_d_fake + gamma * loss_d_fake_e

        vars_d = discriminator.trainable_variables
        grads = tape.gradient(d_loss, vars_d)
        disc_optimizer.apply_gradients(zip(grads, vars_d))
        return d_loss


    # 6. Training Loop
    print('Start Embedding Network Training')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # 注意：TF2 模型处理变长序列最好是 padding 好的，utils.batch_generator 返回的 X_mb 通常是 list
        # 需要转为 tensor/numpy array，形状 (Batch, Seq, Dim)
        # 这里假设 ori_data 已经是 padding 好的矩形数组
        step_e_loss = train_embedder(np.array(X_mb).astype(np.float32))

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {np.sqrt(step_e_loss):.4f}')
    print('Finish Embedding Network Training')

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        step_g_loss_s = train_supervisor(np.array(X_mb).astype(np.float32), np.array(Z_mb).astype(np.float32))

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {np.sqrt(step_g_loss_s):.4f}')
    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')
    for itt in range(iterations):
        # 1. Generator Training (run twice)
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            # Train Generator
            step_g_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(
                np.array(X_mb).astype(np.float32),
                np.array(Z_mb).astype(np.float32)
            )

            # Train Embedder (Jointly)
            step_e_loss_t0 = train_embedder_joint(
                np.array(X_mb).astype(np.float32),
                np.array(Z_mb).astype(np.float32)
            )

        # 2. Discriminator Training
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # Train Discriminator
        step_d_loss = train_discriminator(
            np.array(X_mb).astype(np.float32),
            np.array(Z_mb).astype(np.float32)
        )

        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, d_loss: {step_d_loss:.4f}, '
                  f'g_loss_u: {step_g_loss_u:.4f}, '
                  f'g_loss_s: {np.sqrt(step_g_loss_s):.4f}, '
                  f'g_loss_v: {step_g_loss_v:.4f}, '
                  f'e_loss_t0: {np.sqrt(step_e_loss_t0):.4f}')
    print('Finish Joint Training')

    # 7. Synthetic data generation

    # 获取需要生成的数量
    target_gen_num = parameters.get('n_samples_to_generate', no)

    # 向上取整计算批次
    num_iter = int(np.ceil(target_gen_num / batch_size))
    generated_data = []

    for idx in range(num_iter):
        # 随机采样时间长度 (保持与原数据一致的分布)
        temp_idx = np.random.choice(no, batch_size, replace=True)
        T_mb = [ori_time[i] for i in temp_idx]

        # 生成噪声 Z
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # 前向传播生成数据 (Native TF2 call)
        # Z -> Generator -> E -> Supervisor -> H_hat -> Recovery -> X_hat
        e_hat = generator(np.array(Z_mb).astype(np.float32))
        h_hat = supervisor(e_hat)
        x_hat = recovery(h_hat)

        generated_data_curr = x_hat.numpy() # 转回 numpy

        for i in range(batch_size):
            # 截取有效长度
            curr_len = T_mb[i]
            temp = generated_data_curr[i, :curr_len, :]
            generated_data.append(temp)

    # 截断到目标数量
    generated_data = generated_data[:target_gen_num]
    generated_data = np.array(generated_data)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
