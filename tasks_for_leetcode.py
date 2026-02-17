import statistics
import time
import tracemalloc
from functools import wraps


def make_wrapper(method_to_wrap, test_input, n_repeats):
    @wraps(method_to_wrap)
    def wrapper(self, *args, **kwargs):
        actual_args = args if args else test_input
        times, peaks = [], []

        for _ in range(n_repeats):
            tracemalloc.start()
            start = time.perf_counter_ns()
            method_to_wrap(self, *actual_args, **kwargs)
            end = time.perf_counter_ns()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(end - start)
            peaks.append(peak)

        avg_ms = (statistics.mean(times)) / 1_000_000
        avg_kb = (statistics.mean(peaks)) / 1024
        print(f" {method_to_wrap.__name__:<23} │ {avg_ms:10.6f} ms │ {avg_kb:9.3f} KB")
        return True

    return wrapper


def pack_class(test_input: tuple = (), n_repeats: int = 1000):
    def decorator(cls):
        methods = [
            n
            for n in cls.__dict__
            if callable(getattr(cls, n)) and not n.startswith("_")
        ]

        for name in methods:
            setattr(cls, name, make_wrapper(getattr(cls, name), test_input, n_repeats))

        class Wrapped(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                print(f"\n=== BENCHMARK (N={n_repeats}) | INPUT: {test_input} ===")
                print(f"{'METHOD':<26} │ {'AVG TIME':<14} │ {'AVG PEAK RAM'}")
                print("─" * 70)

                for name in methods:
                    getattr(self, name)()

        return Wrapped

    return decorator


@pack_class(test_input=(43261596,), n_repeats=10000)
class Solution:
    def method1(self, n: int) -> int:
        return 1 + n

    def method2(self, n: int) -> int:
        return 50 + n


if __name__ == "__main__":
    Solution()
