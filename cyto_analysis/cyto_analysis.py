import numpy as np
import pandas as pd
import os
import scipy.optimize as sopt
import scipy.linalg as slin
from scipy.special import expit as sigmoid
from notears import utils
import time
from datetime import timedelta



def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.1):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# Load the cytometry datasets
def load_cyto_data():
    """Load cytometry data and target files into NumPy arrays"""
    
     # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths to the CSV files
    data_path = os.path.join(script_dir, 'cyto_full_data.csv')
    target_path = os.path.join(script_dir, 'cyto_full_target.csv')
    
    print(f"Looking for data file at: {data_path}")
    print(f"Looking for target file at: {target_path}")
    
    # Check if files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")
    

    # Load the data file
    data_df = pd.read_csv(data_path)
    print(f"Data shape: {data_df.shape}")
    print(f"Data columns: {list(data_df.columns)}")
    
    # Load the target file
    target_df = pd.read_csv(target_path)
    print(f"Target shape: {target_df.shape}")
    print(f"Target columns: {list(target_df.columns)}")
    
    protein_names = list(data_df.columns)
    adjacency_df = pd.DataFrame(0, index = protein_names, columns=protein_names)
    for _, row in target_df.iterrows():
        cause = row.iloc[0]
        effect = row.iloc[1]
        adjacency_df.at[cause, effect] = 1

    print(f"\nAdjacency Matrix (DataFrame):")
    print(adjacency_df)
    
    # Convert to NumPy arrays
    X = data_df.values
    adjacency_matrix = adjacency_df.values
    
    return X, adjacency_matrix

if __name__ == '__main__':    
    start_time_total  = time.time()
    np.random.seed(0)
    bootstrap_samples = 30
    n_rows = 10
    w_threshold = 0.1    # Create detailed_results folder
    detailed_results_dir = 'detailed_results'
    os.makedirs(detailed_results_dir, exist_ok=True)


    ### Load the datasets
    X, B_true = load_cyto_data()
    print("Original dataset shape:", X.shape)
    # Randomly sample 50 rows

    selected_rows = np.random.choice(X.shape[0], n_rows, replace=False)
    X = X[selected_rows, :]    
    print("Sampled dataset shape:", X.shape)
    print(f"\nDatasets loaded successfully!")
    print(f"X (features): {X.shape}")
    print(f"Adjacency matrix: {B_true.shape}")

    ### Estimate dag (logistic error)
    print("\nstarting Alg: NOTEARS")
    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    print("Finished running Alg: NOTEARS")    

    ### Accuracies: compare stat with true (compare with weights set to 1 and without set to 1)    # Calculate accuracies for continuous W_est
    acc_w_cont = utils.count_accuracy(B_true, W_est != 0)
    
    # Create binary version of W_est (set non-zero values to 1)
    W_est_binary = W_est.copy()
    W_est_binary[W_est_binary != 0] = 1
    
    # Calculate accuracies for binary W_est
    acc_w_binary = utils.count_accuracy(B_true, W_est_binary != 0)

    # Method 2: Bootstrap estimation
    W_est_bootstrapped = []
    for _ in range(bootstrap_samples):
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        subsample = X[indices]
        W_est_bootstrapped.append(notears_linear(X, lambda1=0.1, loss_type="l2"))
    
    # Stack and average
    W_stack = np.stack(W_est_bootstrapped)
    W_mean = np.mean(W_stack, axis=0)
    W_mean[np.abs(W_mean) < w_threshold] = 0


    # Evaluate bootstrapped model
    acc_with_bootstrap = utils.count_accuracy(B_true, W_mean != 0)

    ### Print Weight Matrices
    print(f"\n===== Estimated Weight Matrix continous values (W_est) =====")
    print(W_est)
    print(f"\n===== Estimated Weight Matrix Binary mapped values (W_est_binary) =====")
    print(W_est_binary)
    print(f"\n===== Estimated Weight Matrix Bootstrapped (W_est_bootstrapped) =====")
    print(W_mean)
    #Print Accuracies
    print("Accuracies W_Est continuous", acc_w_cont)
    print("Accuracies W_est_binary", acc_w_binary)
    print("Accuracies W_Est bootstrapped", acc_with_bootstrap)

    # Save weight matrices
    np.savetxt(os.path.join(detailed_results_dir, "W_est_binary.csv"), W_est_binary, delimiter=',')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_bootstrapped.csv"), W_mean, delimiter=',')
    np.savetxt(os.path.join(detailed_results_dir, "W_est_continuous.csv"), W_est, delimiter=',')

    # Save accuracies as CSV files
    pd.DataFrame([acc_w_cont]).to_csv(os.path.join(detailed_results_dir, "accuracies_continuous.csv"), index=False)
    pd.DataFrame([acc_w_binary]).to_csv(os.path.join(detailed_results_dir, "accuracies_binary.csv"), index=False)
    pd.DataFrame([acc_with_bootstrap]).to_csv(os.path.join(detailed_results_dir, "accuracies_bootstrapped.csv"), index=False)
    ### Total Time 
    total_time = time.time() - start_time_total
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Total execution time: {total_time_str} (format: H:MM:SS)")  