from heart.model_run.conv1d import setup_cnn

# if hc.executable == "cnn" :
executable_models = {"cnn": setup_cnn, "autoencoder": ...}
executable_models["cnn"]()
