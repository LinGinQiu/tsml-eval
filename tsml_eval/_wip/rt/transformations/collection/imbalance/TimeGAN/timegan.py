"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py (Upgraded to TensorFlow 2.x with Auto-Save/Load)

Note: Use original data as training set to generater synthetic data (time-series)
"""

import tensorflow as tf
import numpy as np
from tsml_eval._wip.rt.transformations.collection.imbalance.TimeGAN.utils import extract_time, random_generator, batch_generator
import os
import socket

# ----------------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------------
hostname = socket.gethostname()
is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname
if is_mac:
    # Mac 上强制使用 CPU 防止死锁
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

    # 获取数据集名称和随机种子用于生成保存路径
    # 注意：请确保 _timegan.py 调用时传入了这两个参数
    dataset_name = parameters.get('dataset_name', 'default_dataset')
    random_state = parameters.get('random_state', 0)

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
            model.add(RNN_Layer(units=hidden_dim, return_sequences=True, activation=activation))
        model.add(tf.keras.layers.Dense(units=output_dim, activation=output_activation))
        return model

    # Define the 5 main components
    embedder = make_rnn_model(dim, hidden_dim, num_layers, output_activation='sigmoid', name='embedder')
    recovery = make_rnn_model(hidden_dim, dim, num_layers, output_activation='sigmoid', name='recovery')
    generator = make_rnn_model(dim, hidden_dim, num_layers, output_activation='sigmoid', name='generator')
    supervisor = make_rnn_model(hidden_dim, hidden_dim, num_layers - 1, output_activation='sigmoid', name='supervisor')
    discriminator = make_rnn_model(hidden_dim, 1, num_layers, output_activation=None, name='discriminator')

    # 3. Optimizers
    embedder0_optimizer = tf.keras.optimizers.Adam()
    embedder_optimizer = tf.keras.optimizers.Adam()
    gen_optimizer = tf.keras.optimizers.Adam()
    disc_optimizer = tf.keras.optimizers.Adam()
    gs_optimizer = tf.keras.optimizers.Adam()

    # 4. Loss Functions
    mse = tf.keras.losses.MeanSquaredError()
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 5. Training Steps (Defined as closures)
    # 注意：在 Mac 上如果遇到死锁，请保持 @tf.function 注释状态；在 Linux Server 上可以打开以加速

    # @tf.function
    def train_embedder(x):
        with tf.GradientTape() as tape:
            h = embedder(x)
            x_tilde = recovery(h)
            loss_e0 = 10 * tf.sqrt(mse(x, x_tilde))
        vars_e = embedder.trainable_variables + recovery.trainable_variables
        grads = tape.gradient(loss_e0, vars_e)
        embedder0_optimizer.apply_gradients(zip(grads, vars_e))
        return loss_e0

    # @tf.function
    def train_supervisor(x, z):
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_hat_supervise = supervisor(h)
            loss_s = mse(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        vars_s = supervisor.trainable_variables
        grads = tape.gradient(loss_s, vars_s)
        gs_optimizer.apply_gradients(zip(grads, vars_s))
        return loss_s

    # @tf.function
    def train_generator(x, z):
        with tf.GradientTape() as tape:
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            h_hat_supervise = supervisor(embedder(x))
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            loss_u = bce_logits(tf.ones_like(y_fake), y_fake)
            loss_u_e = bce_logits(tf.ones_like(y_fake_e), y_fake_e)
            loss_s = mse(embedder(x)[:, 1:, :], h_hat_supervise[:, :-1, :])

            x_hat = recovery(h_hat)
            x_mean, x_var = tf.nn.moments(x, axes=[0])
            x_hat_mean, x_hat_var = tf.nn.moments(x_hat, axes=[0])
            loss_v1 = tf.reduce_mean(tf.abs(tf.sqrt(x_hat_var + 1e-6) - tf.sqrt(x_var + 1e-6)))
            loss_v2 = tf.reduce_mean(tf.abs(x_hat_mean - x_mean))
            loss_v = loss_v1 + loss_v2
            g_loss = loss_u + gamma * loss_u_e + 100 * tf.sqrt(loss_s) + 100 * loss_v

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        grads = tape.gradient(g_loss, vars_g)
        gen_optimizer.apply_gradients(zip(grads, vars_g))
        return g_loss, loss_u, loss_s, loss_v

    # @tf.function
    def train_embedder_joint(x, z):
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
            h = embedder(x)
            y_real = discriminator(h)
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)
            loss_d_real = bce_logits(tf.ones_like(y_real), y_real)
            loss_d_fake = bce_logits(tf.zeros_like(y_fake), y_fake)
            loss_d_fake_e = bce_logits(tf.zeros_like(y_fake_e), y_fake_e)
            d_loss = loss_d_real + loss_d_fake + gamma * loss_d_fake_e
        vars_d = discriminator.trainable_variables
        grads = tape.gradient(d_loss, vars_d)
        disc_optimizer.apply_gradients(zip(grads, vars_d))
        return d_loss

    # ----------------------------------------------------------------
    # 6. Training Logic (With Save/Load Check)
    # ----------------------------------------------------------------

    # 构造保存路径
    if is_mac:
        base_save_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/local/saved_models/TimeGAN/'
    else:
        base_save_path = '/scratch/cq2u24/saved_models/TimeGAN/'

    save_dir = os.path.join(base_save_path, f"{dataset_name}_{random_state}")
    os.makedirs(save_dir, exist_ok=True)

    # 定义所有需要保存的网络组件
    networks = {
        'embedder': embedder,
        'recovery': recovery,
        'generator': generator,
        'supervisor': supervisor,
        'discriminator': discriminator
    }

    # 检查是否所有模型权重都已存在
    # [修改前] ... f"{name}.h5" ...
    # [修改后] 改为 .weights.h5
    all_weights_exist = all(os.path.exists(os.path.join(save_dir, f"{name}.weights.h5")) for name in networks)

    do_training = True

    if all_weights_exist:
        print(f"\n[TimeGAN] Found existing trained model at: {save_dir}")
        try:
            # 尝试加载权重
            print("[TimeGAN] Loading weights...")
            for name, net in networks.items():
                # [修改前] ... f"{name}.h5"
                # [修改后] 改为 .weights.h5
                net.load_weights(os.path.join(save_dir, f"{name}.weights.h5"))
            print("[TimeGAN] Weights loaded successfully. Skipping training.")
            do_training = False
        except Exception as e:
            print(f"[TimeGAN] Error loading weights: {e}. Will restart training.")
            do_training = True
    else:
        print(f"\n[TimeGAN] No existing model found at: {save_dir}. Starting training...")

    # 如果需要训练 (模型不存在 或 加载失败)
    if do_training:
        print('Start Embedding Network Training')
        for itt in range(iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            step_e_loss = train_embedder(np.array(X_mb).astype(np.float32))
            if itt % 100 == 0:
                print(f'step: {itt}/{iterations}, e_loss: {np.sqrt(step_e_loss):.4f}')
        print('Finish Embedding Network Training')

        print('Start Training with Supervised Loss Only')
        for itt in range(iterations):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            step_g_loss_s = train_supervisor(np.array(X_mb).astype(np.float32), np.array(Z_mb).astype(np.float32))
            if itt % 100 == 0:
                print(f'step: {itt}/{iterations}, s_loss: {np.sqrt(step_g_loss_s):.4f}')
        print('Finish Training with Supervised Loss Only')

        print('Start Joint Training')
        for itt in range(iterations):
            # Generator training (twice)
            for _ in range(2):
                X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                train_generator(np.array(X_mb).astype(np.float32), np.array(Z_mb).astype(np.float32))
                train_embedder_joint(np.array(X_mb).astype(np.float32), np.array(Z_mb).astype(np.float32))

            # Discriminator training
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            step_d_loss = train_discriminator(np.array(X_mb).astype(np.float32), np.array(Z_mb).astype(np.float32))

            if itt % 100 == 0:
                # 为了日志简洁，这里只打印 d_loss，如果需要其他 loss 可自行添加
                print(f'step: {itt}/{iterations}, d_loss: {step_d_loss:.4f}')
        print('Finish Joint Training')

        # 训练完成后保存
        print(f"[TimeGAN] Saving model weights to: {save_dir}")
        for name, net in networks.items():
            # [修改前] net.save_weights(os.path.join(save_dir, f"{name}.h5"))
            # [修改后] 必须是 .weights.h5
            net.save_weights(os.path.join(save_dir, f"{name}.weights.h5"))

    # ----------------------------------------------------------------
    # 7. Synthetic data generation
    # ----------------------------------------------------------------

    # 获取需要生成的数量
    target_gen_num = parameters.get('n_samples_to_generate', no)

    # 向上取整计算批次
    num_iter = int(np.ceil(target_gen_num / batch_size))
    generated_data = []

    for idx in range(num_iter):
        temp_idx = np.random.choice(no, batch_size, replace=True)
        T_mb = [ori_time[i] for i in temp_idx]
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # 前向传播生成数据 (Native TF2 call)
        # Z -> Generator -> E -> Supervisor -> H_hat -> Recovery -> X_hat
        e_hat = generator(np.array(Z_mb).astype(np.float32))
        h_hat = supervisor(e_hat)
        x_hat = recovery(h_hat)

        generated_data_curr = x_hat.numpy()

        for i in range(batch_size):
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
