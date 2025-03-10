md

import numpy as np


def print_matrix(matrix):
    for row in matrix:
        print([round(elem, 4) for elem in row])


md


# Update value of X on every iteration

def simple_iteration(equations, soll, tolerance=1e-6, max_iterations=20):
    n = len(equations)
    x = [0] * n

    # Construct matrices A* and a*
    A = [[0 if i == j else round(-equations[i][j] / equations[i][i], 4) for j in range(n)] for i in range(n)]
    a = [round(soll[i] / equations[i][i], 4) for i in range(n)]

    print("Matrix A*:")
    print_matrix(A)
    print("Vector a*:", [round(val, 4) for val in a])

    for itr in range(max_iterations):
        new_x = [0] * n
        for i in range(n):
            new_x[i] = round(a[i] + sum(A[i][j] * x[j] for j in range(n)), 4)

        delta = round(max(abs(new_x[i] - x[i]) for i in range(n)), 4)

        print(f"Iteration {itr + 1}: {new_x}, Δ = {delta}")

        if delta < tolerance:
            print("Delta < tolerance!")
            return new_x

        x = new_x

    print("Reached to the maximum number of iterations.")
    return x


md


# Update value of X on every calculation

def simple_iteration_with_updates(equations, soll, tolerance=1e-6, max_iterations=20):
    n = len(equations)
    x = [0] * n

    # Construct matrices A* and a*
    A = [[0 if i == j else round(-equations[i][j] / equations[i][i], 4) for j in range(n)] for i in range(n)]
    a = [round(soll[i] / equations[i][i], 4) for i in range(n)]

    print("Matrix A*:")
    print_matrix(A)
    print("Vector a*:", [round(val, 4) for val in a])

    for itr in range(max_iterations):
        new_x = x.copy()
        for i in range(n):
            new_x[i] = round(a[i] + sum(A[i][j] * new_x[j] for j in range(n)), 4)
        delta = round(max(abs(new_x[i] - x[i]) for i in range(n)), 4)

        print(f"Iteration {itr + 1}: {new_x}, Δ = {delta}")

        if delta < tolerance:
            print("Delta < tolerance!")
            return new_x

        x = new_x

    print("Reached to the maximum number of iterations.")
    return x


def check_positive_defininte(matrix):
    matrix = np.array(matrix)
    if np.array_equal(matrix, matrix.T):
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)
    else:
        print("Matrix is not symmetric.")
        return False


def simple_iterations_unobvious(equations, c_matrix, tao, soll, tolerance=1e-6, max_iterations=20):
    a = soll
    c_matrix = np.array(c_matrix)
    equations = np.array(equations)
    if not check_positive_defininte(equations) or not check_positive_defininte(c_matrix):
        print("Matrix is not positive definite.")
        return
    m = 2 * c_matrix - tao * equations
    while not check_positive_defininte(m):
        tao = round(np.random.rand(), 1)
        m = 2 * c_matrix - tao * equations
    tao_c_inv = np.dot(tao, np.linalg.inv(c_matrix))
    ab = np.dot(tao_c_inv, equations)
    cd = np.dot(tao_c_inv, a)
    x = [0] * (len(equations))
    I = np.eye(len(equations))
    for itr in range(1, max_iterations + 1):
        new_x = x.copy()
        new_x = np.round(np.dot(I - ab, x) + cd, 4)
        delta = round(max(abs(new_x - x)), 4)
        if (itr + 1) % 1 == 0 or itr + 1 == 1:
            print(f"Iteration {itr + 1}: {new_x}, delta = {delta}")
        if delta < tolerance:
            print("Delta < tolerance!")
            # print(f"Iteration {itr + 1}: {new_x}, delta = {delta}")
            return new_x
        x = new_x


equations = [[4, 1, -1],
             [1, 2, 0],
             [-1, 0, 3]]

c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]

tao = round(np.random.rand(), 1)
tao = 0.1
soll = [7, 0, -2]
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[4, 1, -1],
             [1, 2, 0],
             [-1, 0, 3]]

c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]

tao = round(np.random.rand(), 1)
tao = 0.1
soll = [7, 0, -2]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])


def gauss_jordan_update_with_condition(equations, soll):
    n = len(equations)
    matrix = [equations[i] + [soll[i]] for i in range(n)]  # Augmented matrix [A | b]

    print("Initial Augmented Matrix:")
    print_matrix(matrix)

    for index, eq in enumerate(equations):
        k = eq.copy()  # Copy the row
        k.pop(index)  # Remove the diagonal element

        # Check if the diagonal element is greater than the sum of the rest of the row
        if eq[index] > sum(k):
            print(f"Skipping Gauss-Jordan on row {index} as diagonal element is greater than sum of rest")
            continue

        # Apply Gauss-Jordan elimination if the condition is not met
        print(f"Applying Gauss-Jordan to row {index}")
        pivot = matrix[index][index]

        # Normalize the pivot row (divide by the diagonal element to make it 1)
        for j in range(n + 1):  # Loop over all columns (including the solution part)
            matrix[index][j] /= pivot

        # Eliminate the column above and below the pivot
        for i in range(n):
            if i != index:
                factor = matrix[i][index]
                for j in range(n + 1):  # Loop over all columns (including the solution part)
                    matrix[i][j] -= factor * matrix[index][j]

    print("\nFinal Augmented Matrix:")
    print_matrix(matrix)
    print("\n")

    # Extract the solution vector from the last column
    solution = [matrix[i][n] for i in range(n)]

    # Remove the last column to return the updated matrix without the solution part
    updated_matrix = [row[:n] for row in matrix]

    return solution, updated_matrix


equations = [[11, 1.8, 2.7],
             [4.8, 37, 4.1],
             [2.8, 2.1, 33]]
soll = [32, 57, 8]

soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])

equations = [[2.7, 3.3, 1.3],
             [3.5, -1.7, 2.8],
             [4.1, 5.8, -1.7]]
soll = [2.1, 1.7, 0.8]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])

equations = [[1.7, 2.8, 1.9],
             [2.1, 3.4, 1.8],
             [4.2, -1.7, 1.3]]
soll = [0.7, 1.1, 2.8]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]
tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[3.1, 2.8, 1.9],
             [1.9, 3.1, 2.1],
             [7.5, 3.8, 4.8]]
soll = [0.2, 2.1, 5.6]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]
tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[9.1, 5.6, 7.8],
             [3.8, 5.1, 2.8],
             [4.1, 5.7, 1.2]]
soll = [9.8, 6.7, 5.8]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]
tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[5.4, -2.3, 3.4],
             [4.2, 1.7, -2.3],
             [3.4, 2.4, 7.4]]
soll = [-3.5, 2.7, 1.9]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]
tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[2.7, 0.9, -1.5],
             [4.5, -2.8, 6.7],
             [5.1, 3.7, -1.4]]
soll = [3.5, 2.6, -0.14]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]
tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[4, 1, -1],
             [1, 2, 0],
             [-1, 0, 3]]

c_matrix = [[0.5, 0.1, 0.1],
            [0.1, 0.5, 0],
            [0.1, 0, 0.5]]

tao = round(np.random.rand(), 1)
tao = 0.1
soll = [7, 0, -2]
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[1.7, 2.8, 1.9],
             [2.1, 3.4, 1.8],
             [4.2, -1.7, 1.3]]
soll = [0.7, 1.1, 2.8]
soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")

tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[3.6, 1.8, -4.7],
             [2.7, -3.6, 1.9],
             [1.5, 4.5, 3.3]]
soll = [3.8, 0.4, -1.6]
# soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")

tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

equations = [[3.8, 4.1, -2.3],
             [-2.1, 3.9, -5.8],
             [1.8, 1.1, -2.1]]
soll = [4.8, 3.3, 5.8]
# soll, equations = gauss_jordan_update_with_condition(equations, soll)

print("Iterate and update x after every iteration")
solution = simple_iteration(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")
print("Iterate and update x after every calculation")
solution = simple_iteration_with_updates(equations, soll, tolerance=0.001)
print("\nFinal solution:", [round(val, 4) for val in solution])
print("\n------------------------------------\n")

tao = round(np.random.rand(), 1)
a = simple_iterations_unobvious(equations, c_matrix, tao, soll)

