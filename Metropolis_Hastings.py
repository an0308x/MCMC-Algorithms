import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x, mu=0, sigma=1):
    """Our target distribution: a normal distribution."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def proposal_distribution(x, sigma=0.5):
    """Our proposal distribution: a normal distribution centered at x."""
    return np.random.normal(x, sigma)

def metropolis_hastings(n_iterations, initial_state):
    """Metropolis-Hastings algorithm."""
    current_state = initial_state
    samples = [current_state]
    
    for _ in range(n_iterations):
        proposed_state = proposal_distribution(current_state)
        
        acceptance_ratio = target_distribution(proposed_state) / target_distribution(current_state)
        
        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
    
    return samples

# Run the MCMC
n_iterations = 100000
initial_state = 0
samples = metropolis_hastings(n_iterations, initial_state)

# Plot the results
plt.figure(figsize=(10, 5))
plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title("MCMC Samples vs. True Distribution")
plt.xlabel("Value")
plt.ylabel("Density")

# Plot the true distribution for comparison
x = np.linspace(-4, 4, 100)
plt.plot(x, target_distribution(x), 'r-', lw=2, label='True Distribution')
plt.legend()
plt.show()

print(f"Estimated mean: {np.mean(samples)}")
print(f"Estimated standard deviation: {np.std(samples)}")
