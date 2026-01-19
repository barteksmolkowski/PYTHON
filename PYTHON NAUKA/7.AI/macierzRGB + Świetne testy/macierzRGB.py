from PIL import Image, UnidentifiedImageError
import random
import numpy as np
from itertools import product
from functools import wraps
import os

class MatrixRGB:
    def __init__(self, data):
        if data is None or not isinstance(data, list) or len(data) == 0:
            raise ValueError("Data is empty or invalid.")
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix should be a list of lists.")
        self.data = data
        self.height = len(self.data)
        self.width = len(self.data[0])

    @staticmethod
    def show_matrices(matrix_list, count):
        for matrix in matrix_list:
            for i, row in enumerate(matrix):
                print(row)
                if i + 1 == count:
                    break
            print("\n")

    @staticmethod
    def random_matrices_with_tuples(num_matrices, count, x, y, tuple_length, low, high):
        matrices = np.empty((num_matrices, count, x, y), dtype=object)
        for idx in product(range(num_matrices), range(count), range(x), range(y)):
            tuple_weight = tuple(random.randint(low, high) for _ in range(tuple_length))
            matrices[idx] = tuple_weight[0] if len(tuple_weight) == 1 else tuple_weight
        return matrices

class Multiplication:
    @staticmethod
    def matrix_checking(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            A, B = args[0], args[1]

            if not (isinstance(A, list) and isinstance(B, list)):
                raise TypeError("Both matrices must be 2D or 3D lists.")

            def get_shape(mat):
                shape = []
                current = mat
                while isinstance(current, list):
                    shape.append(len(current))
                    current = current[0] if current else []
                return shape

            shapeA = get_shape(A)
            shapeB = get_shape(B)

            if shapeA != shapeB:
                raise ValueError(f"Matrices have different shapes: {shapeA} vs {shapeB}")

            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    @matrix_checking
    def dot_product(A, B):
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                el1 = A[i][j]
                el2 = B[i][j]
                if isinstance(el1, (list, tuple)) and isinstance(el2, (list, tuple)):
                    new_el = tuple(a * b for a, b in zip(el1, el2))
                else:
                    new_el = el1 * el2
                row.append(new_el)
            result.append(row)
        return result

    @staticmethod
    @matrix_checking
    def matrix_multiplication(A, B):
        result = []
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])

        for i, j in product(range(rows_A), range(cols_B)):
            sum_ = sum(A[i][k] * B[k][j] for k in range(cols_A))
            if j == 0:
                result.append([sum_])
            else:
                result[i].append(sum_)
        return result

class Reading:
    def __init__(self):
        pass

    def convert_to_matrix(self, img: Image.Image):
        if img.mode != "RGB":
            raise ValueError(f"Expected image mode 'RGB', but got '{img.mode}'")
        width, height = img.size
        matrix = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(img.getpixel((x, y)))
            matrix.append(row)
        return matrix

    def from_png(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")
        try:
            img = Image.open(path)
            img = img.convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(f"File '{path}' is not a valid image or is corrupted.")
        matrix = self.convert_to_matrix(img)
        return matrix
