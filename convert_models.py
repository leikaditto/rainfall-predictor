# rainfall_app/convert_models.py

import tensorflow as tf
import os

# Folder containing the original .h5 models
source_folder = "models/"
output_folder = "models/converted/"

os.makedirs(output_folder, exist_ok=True)

# Model filenames
models_to_convert = {
    "model_lstm.h5": "model_lstm",
    "model_gru.h5": "model_gru",
    "model_cnnlstm.h5": "model_cnnlstm"
}

# Convert each model
for h5_name, new_folder in models_to_convert.items():
    h5_path = os.path.join(source_folder, h5_name)
    save_path = os.path.join(output_folder, new_folder)

    print(f"Loading {h5_path}...")
    model = tf.keras.models.load_model(h5_path, compile=False)

    print(f"Saving to {save_path} (SavedModel format)...")
    model.export(save_path)

print("✅ All models converted.")