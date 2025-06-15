import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import tkinter as tk
from tkinter import simpledialog

# --- Определение функций системы ---

# Исходная система F(x, y) = 0
# f1(x, y) = tg(y - x) + x*y - 0.3 = 0
# f2(x, y) = x^2 + y^2 - 1.5 = 0

def original_system(vars):
    """
    Исходная система уравнений в виде F(X) = 0.
    Используется для библиотечной функции fsolve.
    """
    x, y = vars
    f1 = np.tan(y - x) + x * y - 0.3
    f2 = x**2 + y**2 - 1.5
    return [f1, f2]

# --- Графическое определение начального приближения ---

def plot_system():
    """
    Строит графики двух уравнений системы для визуального
    определения начальных приближений корней.
    """
    print("Строим графики для определения начального приближения...")
    
    x_range = np.linspace(-2, 2, 400)
    y_range = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_range, y_range)

    F1 = np.tan(Y - X) + X * Y - 0.3
    F2 = X**2 + Y**2 - 1.5

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, F1, levels=[0], colors='blue')
    plt.contour(X, Y, F2, levels=[0], colors='orange')
    
    plt.title('Графики системы нелинейных уравнений')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(['tg(y-x) + xy - 0.3 = 0', 'x² + y² - 1.5 = 0'])
    print("Посмотрите на график. Точки пересечения синей и оранжевой линий являются решениями.")
    print("Определите примерные координаты, закройте окно с графиком и введите их в диалоговых окнах.")
    plt.show()

# --- Система, преобразованная для итерационных методов ---
def g1_seidel(x, y):
    return y - np.arctan(0.3 - x * y)

def g2_seidel(x, y):
    if 1.5 - x**2 < 0:
        return float('nan') 
    return np.sqrt(1.5 - x**2)

# --- МЕТОД ЯКОБИ ---

def jacobi_method(x0, y0, e):
    """
    Реализация метода Якоби.
    Используется другая, более устойчивая форма приведения системы.
    """
    def g1_jacobi(x, y):
        if 1.5 - y**2 < 0: return float('nan')
        return np.sqrt(1.5 - y**2)

    def g2_jacobi(x, y):
        return x + np.arctan(0.3 - x*y)

    x_prev, y_prev = x0, y0
    iters = 0
    max_iters = 1000

    while iters < max_iters:
        x_next = g1_jacobi(x_prev, y_prev)
        y_next = g2_jacobi(x_prev, y_prev)

        if np.isnan(x_next) or np.isnan(y_next):
             return float('nan'), float('nan'), iters
        
        if np.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2) < e:
            return x_next, y_next, iters
        
        x_prev, y_prev = x_next, y_next
        iters += 1
    
    return x_prev, y_prev, iters

# --- МЕТОД ГАУССА-ЗЕЙДЕЛЯ (с релаксацией) ---

def gauss_seidel_method(x0, y0, e, weight=0.5):
    """
    Реализация метода Гаусса-Зейделя с параметром релаксации (weight).
    """
    x_prev, y_prev = x0, y0
    iters = 0
    max_iters = 1000

    while iters < max_iters:
        x_next_g = g1_seidel(x_prev, y_prev)
        if np.isnan(x_next_g):
             return float('nan'), float('nan'), iters
        
        x_next = (1 - weight) * x_prev + weight * x_next_g
        
        y_next_g = g2_seidel(x_next, y_prev) 
        if np.isnan(y_next_g):
             return float('nan'), float('nan'), iters
        
        y_next = (1 - weight) * y_prev + weight * y_next_g

        if np.sqrt((x_next - x_prev)**2 + (y_next - y_prev)**2) < e:
            return x_next, y_next, iters
            
        x_prev, y_prev = x_next, y_next
        iters += 1
        
    return x_prev, y_prev, iters

# --- МЕТОД НЬЮТОНА ---
def newton_method(x0, y0, e):
    """Реализация метода Ньютона."""
    x, y = x0, y0
    iters = 0
    max_iters = 100

    while iters < max_iters:
        f_vec = original_system([x, y])

        tan_val_sq = np.tan(y - x)**2
        j11 = -(1 + tan_val_sq) + y
        j12 = (1 + tan_val_sq) + x
        j21 = 2 * x
        j22 = 2 * y
        
        jacobian = np.array([[j11, j12], [j21, j22]])
        
        try:
            delta = np.linalg.solve(jacobian, [-f_val for f_val in f_vec])
        except np.linalg.LinAlgError:
            return float('nan'), float('nan'), iters

        x_next = x + delta[0]
        y_next = y + delta[1]
        
        if np.sqrt(delta[0]**2 + delta[1]**2) < e:
            return x_next, y_next, iters
            
        x, y = x_next, y_next
        iters += 1
        
    return x, y, iters

# --- ОКНО ВЫВОДА РЕЗУЛЬТАТОВ ---
def show_results_window(results_string):
    """
    Создает кастомное окно для вывода результатов с моноширинным шрифтом.
    """
    results_window = tk.Toplevel()
    results_window.title("Итоговые результаты")
    results_window.geometry("600x400") # Задаем разумный начальный размер
    results_window.resizable(False, False)

    # Используем Text виджет для вывода, так как он поддерживает моноширинные шрифты
    text_widget = tk.Text(results_window, wrap='word', font=("Courier", 10), padx=10, pady=10)
    text_widget.pack(expand=True, fill='both')
    
    text_widget.insert('1.0', results_string)
    text_widget.config(state='disabled') # Делаем текст нередактируемым

    # Кнопка для закрытия окна
    close_button = tk.Button(results_window, text="Закрыть", command=results_window.destroy)
    close_button.pack(pady=10)
    
    # Делаем окно модальным (блокирует взаимодействие с другими окнами)
    results_window.transient()
    results_window.grab_set()
    results_window.wait_window()


# --- ОСНОВНАЯ ЧАСТЬ ПРОГРАММЫ ---
def main():
    root = tk.Tk()
    root.withdraw() # Скрываем основное окно, оно нам не нужно

    # 1. Показываем график
    plot_system()

    # 2. Запрашиваем ввод данных через диалоговые окна
    x0 = simpledialog.askfloat("Ввод данных", "Введите начальное приближение для x:", initialvalue=1.1, parent=root)
    if x0 is None:
        print("Ввод отменен.")
        return

    y0 = simpledialog.askfloat("Ввод данных", "Введите начальное приближение для y:", initialvalue=0.5, parent=root)
    if y0 is None:
        print("Ввод отменен.")
        return

    e = simpledialog.askfloat("Ввод данных", "Введите точность e:", initialvalue=0.001, parent=root)
    if e is None:
        print("Ввод отменен.")
        return

    # 3. Решение методами
    jacobi_x, jacobi_y, jacobi_iters = jacobi_method(x0, y0, e)
    seidel_x, seidel_y, seidel_iters = gauss_seidel_method(x0, y0, e, weight=0.5)
    newton_x, newton_y, newton_iters = newton_method(x0, y0, e)
    scipy_solution = fsolve(original_system, [x0, y0])

    # 4. Формирование строки с результатами для вывода
    results_list = []
    results_list.append("Исходная система уравнений:")
    results_list.append("tg(y - x) + x*y = 0.3")
    results_list.append("x^2 + y^2 = 1.5\n")
    results_list.append(f"Начальное приближение: x0 = {x0}, y0 = {y0}")
    results_list.append(f"Точность: e = {e}\n")

    # Теперь выравнивание будет работать корректно в окне с моноширинным шрифтом
    header = f"{'Метод':<25}{'Количество итераций':<22}{'Значение корней (x, y)'}"
    results_list.append(header)
    results_list.append("-" * len(header))

    if np.isnan(jacobi_x):
        jacobi_res_str = "Метод не сошелся"
    else:
        jacobi_res_str = f"({jacobi_x:.6f}, {jacobi_y:.6f})"
    results_list.append(f"{'Якоби':<25}{str(jacobi_iters):<22}{jacobi_res_str}")

    if np.isnan(seidel_x):
        seidel_res_str = "Метод не сошелся"
    else:
        seidel_res_str = f"({seidel_x:.6f}, {seidel_y:.6f})"
    results_list.append(f"{'Гаусса-Зейделя (w=0.5)':<25}{str(seidel_iters):<22}{seidel_res_str}")
    
    if np.isnan(newton_x):
        newton_res_str = "Метод не сошелся"
    else:
        newton_res_str = f"({newton_x:.6f}, {newton_y:.6f})"
    results_list.append(f"{'Ньютона':<25}{str(newton_iters):<22}{newton_res_str}")
    
    results_list.append("-" * len(header))
    results_list.append("\n--- Результат стандартной библиотечной функции ---")
    results_list.append(f"scipy.optimize.fsolve: x = {scipy_solution[0]:.6f}, y = {scipy_solution[1]:.6f}")

    final_results_string = "\n".join(results_list)

    # 5. Вывод результатов в кастомное окно tkinter и в консоль
    print("\n--- Итоговые результаты вычислений ---")
    print(final_results_string)
    show_results_window(final_results_string)
    
    # 6. Запись в файл
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(final_results_string)
    print("\nРезультаты также сохранены в файл 'results.txt'")

    root.destroy()

if __name__ == "__main__":
    main()
