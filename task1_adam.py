import time
import numpy as np

def f_optimize(f, df, x0, time_limit):
    start_time = time.time()
    max_iterations = 10000
    grad_tol = 1e-6
    
    # Ensure inputs are numpy arrays for element-wise operations
    x = np.array(x0, dtype=float) 
    gradient = np.array(df(x), dtype=float)
    iteration_count = 0
    
    # --- Adam Hyperparameters ---
    alpha = 0.05      # Learning rate / Step size
    beta1 = 0.9       # Exponential decay rate for the 1st moment estimates
    beta2 = 0.999     # Exponential decay rate for the 2nd moment estimates
    epsilon = 1e-8    # Small constant to prevent division by zero
    
    # --- Initialize Moment Vectors ---
    m = np.zeros_like(x) 
    v = np.zeros_like(x) 
    
    current_time = time.time()
    
    while ((current_time - start_time < time_limit) and (iteration_count < max_iterations)):
        
        # 1. Early stopping if we've reached a local minimum
        if np.linalg.norm(gradient) < grad_tol:
            break
            
        iteration_count += 1
        
        # 2. Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        
        # 3. Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # 4. Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** iteration_count)
        
        # 5. Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** iteration_count)
        
        # 6. Update the current guess
        x = x - alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
        
        # 7. Compute the gradient for the next iteration
        gradient = np.array(df(x), dtype=float)
        
        current_time = time.time()

    return x