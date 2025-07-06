import tensorflow as tf

def load_model():
    print("\U0001F4E6 Loading local model (SavedModel format)...")
    model_path = "model_dir"  # ✅ your model path is correct

    try:
        model = tf.saved_model.load(model_path)
        print("\u2705 Model loaded from local directory.")
        # Print available signatures
        print("Available signatures:", list(model.signatures.keys()))
        if "default" in model.signatures:
            return model.signatures["default"]
        elif len(model.signatures) > 0:
            # Use the first available signature
            first_sig = next(iter(model.signatures))
            print(f"Using first available signature: {first_sig}")
            return model.signatures[first_sig]
        else:
            raise ValueError("No callable signatures found in the loaded model.")
    except Exception as e:
        print("\u274C Failed to load model:", e)
        raise
