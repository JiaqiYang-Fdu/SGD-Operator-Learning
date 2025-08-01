import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA



# ============================
#       Data Loading
# ============================
def load_data(input_path="NavierStokes_inputs.npy", output_path="NavierStokes_outputs.npy"):
    X = np.load(input_path)   # shape: (64, 64, N)
    Y = np.load(output_path)  # shape: (64, 64, N)
    print(f"Data loaded: {X.shape[2]} samples")
    return X, Y



# ============================
#       Data Splitting
# ============================
def split_data(X, Y, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    N = X.shape[2]
    indices = np.arange(N)
    # Optionally shuffle data:
    # np.random.seed(seed)
    # np.random.shuffle(indices)

    train_end = int(train_ratio * N)
    valid_end = train_end + int(valid_ratio * N)
    test_end = valid_end + int(test_ratio * N)

    idx_train = indices[:train_end]
    idx_valid = indices[train_end:valid_end]
    idx_test  = indices[valid_end:test_end]

    X_train, Y_train = X[:, :, idx_train], Y[:, :, idx_train]
    X_valid, Y_valid = X[:, :, idx_valid], Y[:, :, idx_valid]
    X_test, Y_test   = X[:, :, idx_test],  Y[:, :, idx_test]

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)



# ============================
#       Preprocessing
# ============================
def preprocess_data(X, method='pca', n_components=100):
    """
    Apply preprocessing (PCA or flattening).

    Args:
        X: np.ndarray of shape (H, W, N)
        method: 'pca' or 'none'
        n_components: PCA dimensionality

    Returns:
        Transformed data (N, D), fitted transformer, original spatial shape
    """
    N = X.shape[2]
    X_flat = X.reshape(-1, N).T  # shape (N, H*W)

    if method == 'pca':
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_flat)
        return X_transformed, pca, X.shape[:2]
    elif method == 'none':
        return X_flat, None, X.shape[:2]
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    

# =========================
#       Kernel Definitions
# =========================
def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def matern_kernel(x, y, nu=1.5, length_scale=1.0):
    r = np.linalg.norm(x - y)
    if nu == 0.5:
        return np.exp(-r / length_scale)
    elif nu == 1.5:
        sqrt3_r = np.sqrt(3) * r / length_scale
        return (1 + sqrt3_r) * np.exp(-sqrt3_r)
    elif nu == 2.5:
        sqrt5_r = np.sqrt(5) * r / length_scale
        return (1 + sqrt5_r + (5 * r ** 2) / (3 * length_scale ** 2)) * np.exp(-sqrt5_r)
    else:
        raise ValueError("Unsupported ν value. Use 0.5, 1.5, or 2.5.")



# ============================
#       Kernel SGD Algorithm
# ============================
def kernel_sgd(X_train, Y_train, X_test=None, Y_test=None, kernel_type='gaussian',
               kernel_params=None, eta0=1.0, decay=0.0,
               verbose=True, preprocessor=None,
               output_inverse_fn=None, output_shape=None,
               plot_errors=False):
    """
    Kernel SGD with test evaluation.

    Args:
        X_train: shape (N, D)
        Y_train: shape (N, D')
    """

    n_train = X_train.shape[0]
    output_dim = Y_train.shape[1]

    X_hist = []
    Alpha_hist = []
    train_errors = []
    test_errors = []
    test_l2_errors = []
    eval_steps = []

    # Select kernel
    if kernel_type == 'gaussian':
        sigma = kernel_params.get('sigma', 1.0)
        kernel = lambda x, y: gaussian_kernel(x, y, sigma)
    elif kernel_type == 'matern':
        nu = kernel_params.get('nu', 1.5)
        length_scale = kernel_params.get('length_scale', 1.0)
        kernel = lambda x, y: matern_kernel(x, y, nu, length_scale)
    else:
        raise ValueError("Unsupported kernel type")

    # Setup real-time plot
    plot_enabled = plot_errors
    if plot_errors:
        try:
            plt.ion()
            fig, ax = plt.subplots()
            error_line, = ax.plot([], [], label='Relative L2 Error')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Relative L2 Error")
            ax.set_title("Kernel SGD Training Error")
            ax.grid(True)
            ax.legend()
        except Exception as e:
            print(f"[Warning] Plotting disabled due to error: {e}")
            plot_enabled = False

    # Evaluation steps: log-spaced indices
    def log_spaced_indices(a, r, T):
        t_list = []
        t = a
        while t <= T:
            t_list.append(int(t))
            t *= r
        return sorted(set(t_list))

    if X_test is not None and Y_test is not None:
        eval_indices = set(log_spaced_indices(10, 2, n_train - 1) + [n_train - 1])

    # === Training loop ===
    for t in tqdm(range(n_train), disable=not verbose):
        xt = X_train[t]
        yt = Y_train[t]

        # Compute prediction
        if t == 0:
            h_xt = np.zeros_like(yt)
        else:
            h_xt = sum(kernel(xt, xj) * alphaj for xj, alphaj in zip(X_hist, Alpha_hist))

        err = h_xt - yt
        rel_err = np.linalg.norm(err) / np.linalg.norm(yt)
        train_errors.append(rel_err)

        # Update model
        eta_t = eta0 * (t + 1) ** (-decay)
        X_hist.append(xt)
        Alpha_hist.append(-eta_t * err)

        # Evaluate on test set if specified
        if X_test is not None and Y_test is not None and t in eval_indices:
            preds = []
            if preprocessor:
                X_proc = preprocessor.transform(X_test.reshape(-1, X_test.shape[2]).T)
            else:
                X_proc = X_test.reshape(-1, X_test.shape[2]).T

            for x in X_proc:
                hx = sum(kernel(x, xj) * alphaj for xj, alphaj in zip(X_hist, Alpha_hist))
                if output_inverse_fn:
                    hx = output_inverse_fn(hx)
                if output_shape:
                    hx = hx.reshape(output_shape)
                preds.append(hx)

            preds = np.array(preds)
            rel_l2_errors = [np.linalg.norm(preds[i] - Y_test[:, :, i]) / np.linalg.norm(Y_test[:, :, i]) for i in range(len(preds))]
            abs_l2_errors = [np.linalg.norm(preds[i] - Y_test[:, :, i]) ** 2 for i in range(len(preds))]
            test_errors.append(np.mean(rel_l2_errors))
            test_l2_errors.append(np.mean(abs_l2_errors) * 4 * np.pi ** 2 / 64 / 64)
            eval_steps.append(t)


        # Update plot every 100 iterations
        if plot_enabled and (t + 1) % 100 == 0:
            try:
                error_line.set_data(range(len(train_errors)), train_errors)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.1)
            except Exception as e:
                print(f"[Warning] Error during live plotting: {e}. Plotting will be disabled.")
                plot_enabled = False

    # Keep the final plot open
    if plot_enabled:
        try:
            plt.ioff()
            plt.show()
        except Exception as e:
            print(f"[Warning] Final plotting skipped due to error: {e}")

    # Prediction function
    def predict(X_raw):
        # Step 1: preprocess X if needed
        if preprocessor:
            X_proc = preprocessor.transform(X_raw.reshape(-1, X_raw.shape[2]).T)
        else:
            X_proc = X_raw.reshape(-1, X_raw.shape[2]).T

        # Step 2: kernel prediction
        preds = []
        for x in X_proc:
            hx = sum(kernel(x, xj) * alphaj for xj, alphaj in zip(X_hist, Alpha_hist))
            if output_inverse_fn:
                hx = output_inverse_fn(hx)
            if output_shape:
                hx = hx.reshape(output_shape)
            preds.append(hx)
        return np.array(preds)

    return predict, train_errors, (eval_steps, test_errors, test_l2_errors)



# ============================
#       Pipeline
# ============================

# Load and split the dataset
X, Y = load_data()
(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_data(X, Y,train_ratio=0.85, valid_ratio=0, test_ratio=0.15)

# Apply PCA preprocessing
X_train_proc, pca_X, input_shape = preprocess_data(X_train, method='pca', n_components=128)
Y_train_proc, pca_Y, output_shape = preprocess_data(Y_train, method='pca', n_components=128)

# Train the model with kernel SGD
predict_fn, train_errors, (eval_steps, test_errors, test_l2_errors)= kernel_sgd(
    X_train_proc, Y_train_proc, X_test=X_test, Y_test=Y_test,
    kernel_type='matern',
    kernel_params={'nu': 2.5, 'length_scale': 0.7},
    eta0=0.9, decay=0,
    preprocessor=pca_X,
    output_inverse_fn=pca_Y.inverse_transform,
    output_shape=output_shape,
    plot_errors=False
)
 

# last1000_train_error = np.mean(train_errors[-1000:]) if len(train_errors) >= 1000 else np.mean(train_errors)
# print(f"Last1000 Train L2 Error = {last1000_train_error:.4f}")

# def smooth_curve(values, window_size=10):
#     window = np.ones(window_size) / window_size
#     return np.convolve(values, window, mode='valid')

# smoothed_errors = smooth_curve(train_errors, window_size=100)

# plt.plot(smoothed_errors, label='Smoothed Train Error')
# plt.xlabel('Iteration')
# plt.ylabel('Error')
# plt.title('Smoothed Training Error')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Test on the validation set
# Y_valid_pred = predict_fn(X_valid)


# # Compute the relative error
# errors = [np.linalg.norm(Y_valid_pred[i] - Y_valid[:, :, i]) / np.linalg.norm(Y_valid[:, :, i])
#           for i in range(Y_valid.shape[2])]
# print(f"Mean validation relative L2 error: {np.mean(errors):.4f}")



# ============================
#       Save Predictions
# ============================

# Choose a test sample to visualize/save
idx = 0  
x_sample = X_test[:, :, idx:idx+1]  # Shape: (64, 64, 1)
y_true = Y_test[:, :, idx]          # Ground truth
y_pred = predict_fn(x_sample)[0]    # Predicted field

# Save results for later plotting/analysis
np.save("true_field.npy", y_true)
np.save("predicted_field.npy", y_pred)
np.save("eval_steps.npy", eval_steps)
np.save("test_errors.npy", test_errors)
np.save("test_l2_errors", test_l2_errors)

