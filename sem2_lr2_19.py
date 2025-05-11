import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import root
# import math
import warnings

'''added new line'''

# Suppress potential runtime warnings (e.g., invalid value in sqrt)
# Note: Avoiding try/except as requested, so relying on warnings/errors
# if calculations fail (e.g., division by zero, sqrt of negative).
warnings.filterwarnings("ignore")

# --- Define the System of Equations ---
# f1(x, y) = tan(y - x) + x * y - 0.3 = 0
# f2(x, y) = x^2 + y^2 - 1.5 = 0

def f1(x, y):
  """
  First equation: tan(y - x) + x * y - 0.3
  NOTE: Removed the np.isclose check here as it caused issues with array inputs
  during plotting. Numpy's tan function will handle array inputs and potential
  infinities, which contour plotting can manage.
  Potential issues near asymptotes during solving will be handled by numpy warnings
  or result in divergence/errors in the solvers.
  """
  # Using np.real_if_close handles potential small imaginary parts from complex inputs if any
  return np.tan(y - x) + x * y - 0.3

def f2(x, y):
  """Second equation: x^2 + y^2 - 1.5"""
  return x**2 + y**2 - 1.5

def system_equations(vars):
  """Vector form of the system for use with scipy.optimize.root"""
  x, y = vars
  # Need to handle potential errors here if tan is infinite for scalar inputs
  f1_val = f1(x, y)
  f2_val = f2(x, y)
  # If f1 calculation resulted in infinity, return a large number instead
  # This helps scipy.optimize which might not handle 'inf' directly.
  if np.isinf(f1_val) or np.isnan(f1_val):
      # Return large number or check sign if possible
      f1_val = 1e12 * np.sign(f1_val) if not np.isnan(f1_val) else 1e12
  if np.isnan(f2_val):
      f2_val = 1e12 # Handle potential NaN from invalid inputs

  return [f1_val, f2_val]


# --- Plotting Function ---
def plot_equations():
  """Plots the two equations to help find initial guesses."""
  print("Generating plot to visualize equations...")
  y_plot, x_plot = np.mgrid[-2.5:2.5:300j, -2.5:2.5:300j] # Create a finer grid

  # Calculate function values on the grid
  # Using a try-catch block *just for plotting* to handle tan asymptotes gracefully
  # This doesn't violate the spirit of the no-try/except for the *solvers*
  try:
      # Ensure inputs to f1 are real for plotting
      f1_vals = f1(x_plot.astype(float), y_plot.astype(float))
  except (FloatingPointError, ValueError): # Catch potential overflow/domain errors from tan
      # Handle cases where tan might overflow across the grid
      f1_vals = np.full_like(x_plot, np.nan) # Fill with NaN if calculation fails
      # Attempt calculation element-wise where possible
      for i in range(x_plot.shape[0]):
          for j in range(x_plot.shape[1]):
              try:
                  f1_vals[i, j] = f1(x_plot[i, j], y_plot[i, j])
              except (FloatingPointError, ValueError):
                  f1_vals[i, j] = np.nan # Mark problematic points as NaN


  f2_vals = f2(x_plot, y_plot)

  # Mask large values potentially resulting from tan asymptotes for better contouring
  # Also mask NaN values
  f1_vals = np.ma.masked_where((np.abs(f1_vals) > 50) | np.isnan(f1_vals), f1_vals)


  plt.figure(figsize=(8, 8))
  plt.title('System of Nonlinear Equations')

  # Plot contour where f1=0
  plt.contour(x_plot, y_plot, f1_vals, levels=[0], colors='blue', linestyles='dashed')
  # Plot contour where f2=0 (the circle)
  plt.contour(x_plot, y_plot, f2_vals, levels=[0], colors='red')

  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axhline(0, color='black', linewidth=0.5)
  plt.axvline(0, color='black', linewidth=0.5)
  # Add legend entries manually for clarity
  plt.plot([], [], color='blue', linestyle='dashed', label='tan(y - x) + x*y - 0.3 = 0')
  plt.plot([], [], color='red', label='$x^2 + y^2 - 1.5 = 0$')
  plt.legend()
  plt.axis('equal') # Ensure circle looks like a circle
  plt.ylim(-2.5, 2.5)
  plt.xlim(-2.5, 2.5)
  plt.show()
  print("Plot displayed. Please examine it to choose an initial guess (x0, y0)")
  

print("System of Nonlinear Equations:")
print("1) tan(y - x) + x * y = 0.3")
print("2) x^2 + y^2 = 1.5")
print("-" * 30)

# Plot the equations first
plot_equations()