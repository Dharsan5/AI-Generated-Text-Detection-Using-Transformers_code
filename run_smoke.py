import os
import dill as pickle
import numpy as np

def main():
    model_path = os.path.join("model", "model")
    mu_path = os.path.join("model", "mu")
    sigma_path = os.path.join("model", "sigma")

    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    assert os.path.exists(mu_path), f"Mu file not found: {mu_path}"
    assert os.path.exists(sigma_path), f"Sigma file not found: {sigma_path}"

    model = pickle.load(open(model_path, "rb"))
    mu = pickle.load(open(mu_path, "rb"))
    sigma = pickle.load(open(sigma_path, "rb"))

    n = mu.shape[0]
    print(f"Loaded model. Expected feature length: {n}")

    # The trained model expects normalized features. Create a neutral (zero) vector
    # which corresponds to the mean input after normalization.
    sample = np.zeros((1, n))

    try:
        probs = model.predict_proba(sample)[:, 1]
    except Exception as e:
        print("Model prediction failed:", e)
        return

    print("Smoke test prediction probability (AI):", probs[0])

if __name__ == '__main__':
    main()
