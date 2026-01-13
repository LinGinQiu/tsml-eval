import os, warnings
from tsml_eval._wip.rt.transformations.collection.imbalance.TimeVAE.data_utils import (
    inverse_transform_data,
    scale_data,
    load_yaml_file
)
from tsml_eval._wip.rt.transformations.collection.imbalance.TimeVAE.vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)


def run_vae_pipeline(train_data, valid_data=None, vae_type: str='timeVAE', n_samples=100):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    import socket
    hostname = socket.gethostname()
    is_mac = "mac" in hostname.lower() or "CH-Qiu" in hostname  # 你的本机名
    if is_mac:
        yaml_path = '/Users/qiuchuanhang/PycharmProjects/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
    else:
        yaml_path = '/home/cq2u24/tsml-eval/tsml_eval/_wip/rt/transformations/collection/imbalance/TimeVAE/vae/hyperparameters.yaml'
    hyperparameters = load_yaml_file(yaml_path)[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape

    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=1000,
        verbose=1,
    )

    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=n_samples)

    # inverse transformer samples to original scale and save to dir
    inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    generated_samples = inverse_scaled_prior_samples.reshape(n_samples, sequence_length, feature_dim)

    return generated_samples


if __name__ == "__main__":
    # check `/data/` for available datasets
    dataset = "sine_subsampled_train_perc_20"

    # models: vae_dense, vae_conv, timeVAE
    model_name = "timeVAE"

    run_vae_pipeline(dataset, model_name)
