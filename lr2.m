clear all;
clc;

% f1 = @(x,y) tan(y - x) + x.*y - 0.3;
% f2 = @(x,y) x.^2 + y.^2 - 1.5;


disp('Построение графиков.');
figure;
hold on;

h1 = ezplot(f1, [-2 2]);
set(h1, 'Color', 'b');

h2 = ezplot(f2, [-2 2]);
set(h2, 'Color', 'r');

title('Графики системы нелинейных уравнений');
xlabel('x');
ylabel('y');
grid on;
axis equal;
legend('f1: tg(y-x) + xy - 0.3 = 0', 'f2: x^2 + y^2 - 1.5 = 0', 'Location', 'northwest');
hold off;

disp('Поиск решений.');

% функция-обертка, которая принимает вектор v=[x,y]
% и вызывает функции f1(x,y) и f2(x,y),
% возвращая их результаты как единый вектор-столбец.
system_handle = @(v) [f1(v(1), v(2));f2(v(1), v(2))];


% Находим первый корень
initial_guess1 = [1; 0.5];
[root1, ~, exitflag1] = fsolve(system_handle, initial_guess1);

% Находим второй корень
initial_guess2 = [-1; -0.5];
[root2, ~, exitflag2] = fsolve(system_handle, initial_guess2);

% --- Вывод результатов ---
disp('--- Результаты решения системы ---');

if exitflag1 > 0
    fprintf('Найден первый корень:\n x1 = %f\n y1 = %f\n', root1(1), root1(2));
else
    fprintf('Первый корень не найден.\n');
end

fprintf('\n');

if exitflag2 > 0
    fprintf('Найден второй корень:\n x2 = %f\n y2 = %f\n', root2(1), root2(2));
else
    fprintf('Второй корень не найден.\n');
end
