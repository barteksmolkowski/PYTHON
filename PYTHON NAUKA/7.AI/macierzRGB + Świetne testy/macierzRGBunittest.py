import os
import sys
from time import perf_counter
from functools import wraps

from PIL import Image

from macierzRGB import MatrixRGB, Multiplication, Reading

TEST_STATS = {"passed": 0, "failed": 0}

def test(func):
    @wraps(func)
    def wrapper():
        print(f"Sprawdzanie testu: {func.__name__}")
        try:
            func()
            print("V Test zaliczony\n")
        except AssertionError as e:
            print(f"X Błąd asercji: {e}\n")
        except Exception as e:
            print(f"X Nieoczekiwany wyjątek: {type(e).__name__}: {e}\n")
    return wrapper

def expect_exception(expected_exception):
    def decorator(func):
        @wraps(func)
        def wrapper():
            print(f"Sprawdzanie testu: {func.__name__}")
            try:
                func()
                print("X Błąd – wyjątek nie został rzucony\n")
            except expected_exception:
                print("V Test zaliczony (złapano wyjątek)\n")
            except Exception as e:
                print(f"X Zły wyjątek: {type(e).__name__}: {e}\n")
        return wrapper
    return decorator

def count_tests(expected_exception=None):
    def decorator(func):
        @wraps(func)
        def wrapper():
            try:
                func()
            except Exception as e:
                if expected_exception and isinstance(e, expected_exception):
                    TEST_STATS["passed"] += 1
                    print(f"Test {func.__name__} zaliczony (rzucił oczekiwany wyjątek).")
                else:
                    TEST_STATS["failed"] += 1
                    print(f"Test {func.__name__} NIE zaliczony! Błąd: {type(e).__name__}: {e}")
            else:
                if expected_exception:
                    TEST_STATS["failed"] += 1
                    print(f"Test {func.__name__} NIE zaliczony! Nie rzucono oczekiwanego wyjątku {expected_exception.__name__}.")
                else:
                    TEST_STATS["passed"] += 1
                    print(f"Test {func.__name__} zaliczony.")
        return wrapper
    return decorator

def measure_time(func):
    @wraps(func)
    def wrapper():
        start = perf_counter()
        func()
        end = perf_counter()
        print(f"Czas wykonania: {end - start:.6f}s\n")
    return wrapper

def silent(func):
    @wraps(func)
    def wrapper():
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            func()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
    return wrapper

@test
@count_tests()
@measure_time
def test_reading_valid_image():
    reader = Reading()
    img = Image.new("RGB", (3, 3), color=(123, 222, 111))
    path = "temp_test_img.png"
    img.save(path)

    try:
        matrix = reader.from_png(path)
        assert isinstance(matrix, list), "Wynik nie jest listą"
        assert len(matrix) == 3 and len(matrix[0]) == 3, "Zły rozmiar macierzy"
    finally:
        if os.path.exists(path):
            os.remove(path)

@expect_exception(FileNotFoundError)
@count_tests(FileNotFoundError)
@measure_time
def test_reading_invalid_path():
    reader = Reading()
    reader.from_png("plik_nie_istnieje.png")

@expect_exception(ValueError)
@count_tests(ValueError)
@measure_time
def test_reading_invalid_image():
    reader = Reading()
    path = "zly_plik.png"

    with open(path, "w") as f:
        f.write("to nie jest obraz")

    try:
        reader.from_png(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

@test
@count_tests()
@measure_time
def test_random_matrices_with_tuples():
    matrices = MatrixRGB.random_matrices_with_tuples(
        num_matrices=2,
        count=3,
        x=2,
        y=2,
        tuple_length=3,
        low=1,
        high=5
    )
    assert matrices.shape == (2, 3, 2, 2), "Zły kształt macierzy"

@test
@count_tests()
@measure_time
def test_dot_product_valid():
    A = [
        [(1, 2), (3, 4)],
        [(5, 6), (7, 8)]
    ]
    B = [
        [(2, 0), (1, 1)],
        [(0, 1), (1, 0)]
    ]

    expected = [
        [(2, 0), (3, 4)],
        [(0, 6), (7, 0)]
    ]

    result = Multiplication.dot_product(A, B)
    assert result == expected, "Niepoprawny wynik dot_product"

@expect_exception(ValueError)
@count_tests(ValueError)
@measure_time
def test_dot_product_invalid_shape():
    A = [[1, 2], [3, 4]]
    B = [[1, 2, 3], [4, 5, 6]]
    Multiplication.dot_product(A, B)

@test
@count_tests()
@measure_time
def test_matrix_multiplication_valid():
    A = [[1, 2],[3, 4]] # A = [[1, 2],[3, 4]]
    B = [[2, 0],[1, 2]]

    expected = [[4, 4],[10, 8]]

    result = Multiplication.matrix_multiplication(A, B)
    assert result == expected, "Niepoprawny wynik matrix_multiplication"

@expect_exception(ValueError)
@count_tests(ValueError)
@measure_time
def test_matrix_multiplication_invalid_shape():
    A = [[1, 2]]
    B = [[1, 2], [3, 4], [5, 6]]
    Multiplication.matrix_multiplication(A, B)

if __name__ == "__main__":
    print("\n=== ROZPOCZĘCIE TESTÓW ===\n")

    test_reading_valid_image()
    test_reading_invalid_path()
    test_reading_invalid_image()

    test_random_matrices_with_tuples()

    test_dot_product_valid()
    test_dot_product_invalid_shape()

    test_matrix_multiplication_valid()
    test_matrix_multiplication_invalid_shape()

    print(f"Testy zakończone. Zaliczone: {TEST_STATS['passed']}, Nie zaliczone: {TEST_STATS['failed']}")
    print("=== KONIEC TESTÓW ===")
