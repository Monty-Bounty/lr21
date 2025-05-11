import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import warnings

# Suppress potential runtime warnings
warnings.filterwarnings("ignore")

# --- Define the System of Equations ---
# f1(x, y) = tan(y - x) + x * y - 0.3 = 0
# f2(x, y) = x^2 + y^2 - 1.5 = 0

def f1(x, y):
  """
  Первое уравнение: tan(y - x) + x * y - 0.3
  Handles potential tan singularity.
  """
  cos_val = np.cos(y - x)
  if np.abs(cos_val) < 1e-9: 
      return np.sign(np.sin(y - x)) * 1e12 
  tan_val = np.tan(y - x)
  if np.isinf(tan_val):
      return np.sign(tan_val) * 1e12
  return tan_val + x * y - 0.3

def f2(x, y):
  """Второе уравнение: x^2 + y^2 - 1.5"""
  return x**2 + y**2 - 1.5

def system_equations(vars_in):
  """Векторная форма системы для использования с scipy.optimize.root"""
  x, y = vars_in
  f1_val = f1(x, y)
  f2_val = f2(x, y)
  
  if np.isinf(f1_val) or np.isnan(f1_val):
      f1_val = 1e12 * (np.sign(f1_val) if not np.isnan(f1_val) else 1.0)
  if np.isnan(f2_val): 
      f2_val = 1e12
  return [f1_val, f2_val]

# --- Plotting Function ---
def plot_equations():
  """Строит графики двух уравнений для помощи в поиске начальных приближений."""
  print("Генерация графика для визуализации уравнений...")
  y_plot, x_plot = np.mgrid[-2.5:2.5:300j, -2.5:2.5:300j] 
  f1_vals = np.zeros_like(x_plot, dtype=float)
  for i in range(x_plot.shape[0]):
      for j in range(x_plot.shape[1]):
          f1_vals[i,j] = f1(x_plot[i,j], y_plot[i,j])
  f2_vals = f2(x_plot, y_plot)
  f1_vals = np.ma.masked_where((np.abs(f1_vals) > 10) | np.isnan(f1_vals), f1_vals)

  plt.figure(figsize=(8, 8))
  plt.title('Система нелинейных уравнений')
  plt.contour(x_plot, y_plot, f1_vals, levels=[0], colors='blue', linestyles='dashed')
  plt.contour(x_plot, y_plot, f2_vals, levels=[0], colors='red')
  plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
  plt.axhline(0, color='black', lw=0.5); plt.axvline(0, color='black', lw=0.5)
  plt.plot([], [], color='blue', linestyle='dashed', label='$tan(y - x) + x*y - 0.3 = 0$')
  plt.plot([], [], color='red', label='$x^2 + y^2 - 1.5 = 0$')
  plt.legend(); plt.axis('equal'); plt.ylim(-2.5, 2.5); plt.xlim(-2.5, 2.5)
  plt.show()
  print("График отображен. Пожалуйста, изучите его для выбора начального приближения (x0, y0).")

# --- Original Iteration functions G1, G2 ---
def G1(y, x_sign_preference):
    """x_new = x_sign_preference * sqrt(1.5 - y^2)"""
    val_sqrt = 1.5 - y**2
    if val_sqrt < 0: return None
    return x_sign_preference * np.sqrt(val_sqrt)

def G2(x, y_arg):
    """y_new = x + atan(0.3 - x*y_arg)"""
    return x + np.arctan(0.3 - x * y_arg)

# --- Jacobi Method (Original G1, G2) ---
def jacobi_method(x0, y0, epsilon, max_iter=100):
    x_k, y_k = x0, y0
    x_sign_pref = np.sign(x0) if x0 != 0 else 1.0
    for i in range(max_iter):
        x_prev, y_prev = x_k, y_k
        x_kp1 = G1(y_prev, x_sign_pref)
        if x_kp1 is None: return None, None, i + 1, "Ошибка (sqrt отрицательного для x)"
        y_kp1 = G2(x_prev, y_prev)
        if np.isnan(y_kp1) or np.isinf(y_kp1) or np.isnan(x_kp1) or np.isinf(x_kp1):
            return None, None, i + 1, "Ошибка (x или y стал NaN/Inf)"
        x_k, y_k = x_kp1, y_kp1
        if np.sqrt((x_k - x_prev)**2 + (y_k - y_prev)**2) < epsilon:
            return x_k, y_k, i + 1, "Успех"
    return x_k, y_k, max_iter, "Достигнуто макс. итераций"

# --- Gauss-Seidel Method (Original G1, G2) ---
def gauss_seidel_method(x0, y0, epsilon, max_iter=100):
    x_k, y_k = x0, y0
    x_sign_pref = np.sign(x0) if x0 != 0 else 1.0
    for i in range(max_iter):
        x_prev, y_prev = x_k, y_k
        x_kp1 = G1(y_k, x_sign_pref)
        if x_kp1 is None: return None, None, i + 1, "Ошибка (sqrt отрицательного для x)"
        y_kp1 = G2(x_kp1, y_k)
        if np.isnan(y_kp1) or np.isinf(y_kp1) or np.isnan(x_kp1) or np.isinf(x_kp1):
            return None, None, i + 1, "Ошибка (x или y стал NaN/Inf)"
        x_k, y_k = x_kp1, y_kp1
        if np.sqrt((x_k - x_prev)**2 + (y_k - y_prev)**2) < epsilon:
            return x_k, y_k, i + 1, "Успех"
    return x_k, y_k, max_iter, "Достигнуто макс. итераций"

# --- Alternative Iteration functions H1_alt, H2_alt ---
def H1_alt(iter_x, iter_y): # Iteration for x from f1: x_new = (0.3 - tan(y - x)) / y
    if abs(iter_y) < 1e-9: return None, "Деление на ноль в H1_alt"
    cos_val = np.cos(iter_y - iter_x)
    if abs(cos_val) < 1e-9: return None, "Особенность tan в H1_alt (cos~0)"
    tan_val = np.tan(iter_y - iter_x)
    if np.isinf(tan_val) or np.isnan(tan_val): return None, "Результат tan inf/nan в H1_alt"
    return (0.3 - tan_val) / iter_y, "Успех H1"

def H2_alt(iter_x, y_sign_preference): # Iteration for y from f2: y_new = y_sign_pref * sqrt(1.5 - x^2)
    val_sqrt = 1.5 - iter_x**2
    if val_sqrt < 0: return None, "Sqrt отриц. числа в H2_alt"
    if np.isnan(val_sqrt): return None, "Аргумент Sqrt NaN в H2_alt"
    return y_sign_preference * np.sqrt(val_sqrt), "Успех H2"

# --- Jacobi Method (Alternative H1_alt, H2_alt) ---
def jacobi_method_alt(x0, y0, epsilon, max_iter=100):
    x_k, y_k = x0, y0
    y_sign_pref = np.sign(y0) if y0 != 0 else 1.0
    for i in range(max_iter):
        x_prev, y_prev = x_k, y_k
        
        # y_kp1 из x_prev
        y_kp1, status_h2 = H2_alt(x_prev, y_sign_pref)
        if y_kp1 is None: return None, None, i + 1, f"Ошибка H2: {status_h2}"
        
        # x_kp1 из x_prev, y_prev
        x_kp1, status_h1 = H1_alt(x_prev, y_prev)
        if x_kp1 is None: return None, None, i + 1, f"Ошибка H1: {status_h1}"

        if np.isnan(y_kp1) or np.isinf(y_kp1) or np.isnan(x_kp1) or np.isinf(x_kp1):
            return None, None, i + 1, "Ошибка (x/y NaN/Inf Альт.)"
        x_k, y_k = x_kp1, y_kp1
        if np.sqrt((x_k - x_prev)**2 + (y_k - y_prev)**2) < epsilon:
            return x_k, y_k, i + 1, "Успех (Альт.)"
    return x_k, y_k, max_iter, "Достигнуто макс. итераций (Альт.)"

# --- Gauss-Seidel Method (Alternative H1_alt, H2_alt) ---
def gauss_seidel_method_alt(x0, y0, epsilon, max_iter=100):
    x_k, y_k = x0, y0
    y_sign_pref = np.sign(y0) if y0 != 0 else 1.0
    for i in range(max_iter):
        x_prev, y_prev = x_k, y_k

        # Обновляем y используя x_k
        y_kp1, status_h2 = H2_alt(x_k, y_sign_pref)
        if y_kp1 is None: return None, None, i + 1, f"Ошибка H2: {status_h2}"
        
        # Обновляем x используя y_kp1 (новый y) и x_k (старый x)
        x_kp1, status_h1 = H1_alt(x_k, y_kp1) # Используем y_kp1
        if x_kp1 is None: return None, None, i + 1, f"Ошибка H1: {status_h1}"

        if np.isnan(y_kp1) or np.isinf(y_kp1) or np.isnan(x_kp1) or np.isinf(x_kp1):
            return None, None, i + 1, "Ошибка (x/y NaN/Inf Альт.)"
        x_k, y_k = x_kp1, y_kp1
        if np.sqrt((x_k - x_prev)**2 + (y_k - y_prev)**2) < epsilon:
            return x_k, y_k, i + 1, "Успех (Альт.)"
    return x_k, y_k, max_iter, "Достигнуто макс. итераций (Альт.)"

# --- Newton's Method ---
def newton_method(x0, y0, epsilon, max_iter=100):
    x_k, y_k = x0, y0
    for i in range(max_iter):
        F_val = np.array([f1(x_k, y_k), f2(x_k, y_k)])
        if np.isinf(F_val).any() or np.isnan(F_val).any():
             return None, None, i + 1, "Ошибка (F стал NaN/Inf)"
        cos_y_minus_x = np.cos(y_k - x_k)
        if np.abs(cos_y_minus_x) < 1e-9: 
            return None, None, i + 1, "Ошибка (cos(y-x) ~ 0 в Якобиане)"
        sec_sq_term = 1.0 / (cos_y_minus_x**2)
        df1dx = -sec_sq_term + y_k 
        df1dy = sec_sq_term + x_k   
        df2dx = 2 * x_k; df2dy = 2 * y_k
        J = np.array([[df1dx, df1dy], [df2dx, df2dy]])
        det_J = np.linalg.det(J)
        if np.abs(det_J) < 1e-9: 
            return None, None, i + 1, "Ошибка (вырожденный Якобиан)"
        delta = np.linalg.solve(J, -F_val) 
        x_kp1 = x_k + delta[0]; y_kp1 = y_k + delta[1]
        if np.sqrt((x_kp1 - x_k)**2 + (y_kp1 - y_k)**2) < epsilon:
            return x_kp1, y_kp1, i + 1, "Успех"
        x_k, y_k = x_kp1, y_kp1
    return x_k, y_k, max_iter, "Достигнуто макс. итераций"

# --- SciPy Solver ---
def scipy_solver(x0, y0, epsilon_val):
    sol = root(system_equations, [x0, y0], tol=epsilon_val/10.0, method='hybr') 
    if sol.success:
        return sol.x[0], sol.x[1], sol.nfev, f"Успех ({sol.message})"
    else:
        nfev = sol.nfev if hasattr(sol, 'nfev') else 0 
        return None, None, nfev, f"Ошибка ({sol.message})"

# --- Main Execution ---
if __name__ == "__main__":
    print("Система нелинейных уравнений:")
    print("1) tan(y - x) + x * y - 0.3 = 0")
    print("2) x^2 + y^2 - 1.5 = 0")
    print("-" * 40)
    plot_equations()

    try:
        x_initial = float(input("Введите начальное приближение для x (x0): "))
        y_initial = float(input("Введите начальное приближение для y (y0): "))
        epsilon_val = float(input("Введите желаемую точность (например, 0.001): "))
        if epsilon_val <= 0: raise ValueError("Точность должна быть > 0")
    except ValueError as e:
        print(f"Ошибка ввода: {e}. Пожалуйста, введите числовые значения.")
        exit()

    print("-" * 40)
    print(f"Начальное приближение: x0 = {x_initial}, y0 = {y_initial}")
    print(f"Точность epsilon = {epsilon_val}")
    print("-" * 70)
    print(f"{'Метод':<25} | {'Кол-во итераций':<18} | {'x':<15} | {'y':<15} | {'Статус'}")
    print("-" * 70)

    results = []
    methods_to_run = [
        ("Якоби (Ориг.)", jacobi_method),
        ("Гаусса-Зейделя (Ориг.)", gauss_seidel_method),
        ("Якоби (Альт.)", jacobi_method_alt),
        ("Гаусса-Зейделя (Альт.)", gauss_seidel_method_alt),
        ("Ньютона", newton_method),
        ("SciPy optimize.root", scipy_solver)
    ]

    for name, func in methods_to_run:
        if "SciPy" in name: # SciPy's nfev is not exactly iterations
            x_res, y_res, it_res, status_res = func(x_initial, y_initial, epsilon_val)
            iter_label = "Вызовы функции"
        else:
            x_res, y_res, it_res, status_res = func(x_initial, y_initial, epsilon_val, 100) # max_iter=100
            iter_label = "Итерации"
        
        x_str = f"{x_res:.6f}" if x_res is not None else "N/A"
        y_str = f"{y_res:.6f}" if y_res is not None else "N/A"
        # Для SciPy выводим nfev, для остальных - итерации
        iter_count_str = str(it_res)
        
        # Обновляем вывод таблицы для соответствия
        if "SciPy" in name:
             print(f"{name:<25} | {iter_count_str:<18} | {x_str:<15} | {y_str:<15} | {status_res}")
        else:
             print(f"{name:<25} | {iter_count_str:<18} | {x_str:<15} | {y_str:<15} | {status_res}")


    print("-" * 70)



'''
Начальное приближение: x0 = 1.0, y0 = 0.7
Точность epsilon = 0.001
----------------------------------------------------------------------
Метод                     | Кол-во итераций    | x               | y               | Статус
----------------------------------------------------------------------
Якоби (Ориг.)             | 10                 | 1.029219        | 0.663473        | Успех
Гаусса-Зейделя (Ориг.)    | 100                | 1.199107        | 1.200190        | Достигнуто макс. итераций
Якоби (Альт.)             | 6                  | N/A             | N/A             | Ошибка H2: Sqrt отриц. числа в H2_alt
Гаусса-Зейделя (Альт.)    | 5                  | N/A             | N/A             | Ошибка H2: Sqrt отриц. числа в H2_alt
Ньютона                   | 3                  | 1.029406        | 0.663569        | Успех
SciPy optimize.root       | 9                  | 1.029406        | 0.663569        | Успех (The solution converged.)

Начальное приближение: x0 = -0.7, y0 = -1.0
Точность epsilon = 0.001
----------------------------------------------------------------------
Метод                     | Кол-во итераций    | x               | y               | Статус
----------------------------------------------------------------------
Якоби (Ориг.)             | 7                  | N/A             | N/A             | Ошибка (sqrt отрицательного для x)
Гаусса-Зейделя (Ориг.)    | 4                  | N/A             | N/A             | Ошибка (sqrt отрицательного для x)
Якоби (Альт.)             | 16                 | -0.663762       | -1.029598       | Успех (Альт.)
Гаусса-Зейделя (Альт.)    | 9                  | N/A             | N/A             | Ошибка H2: Sqrt отриц. числа в H2_alt
Ньютона                   | 3                  | -0.663569       | -1.029406       | Успех
SciPy optimize.root       | 9                  | -0.663569       | -1.029406       | Успех (The solution converged.)
----------------------------------------------------------------------
'''