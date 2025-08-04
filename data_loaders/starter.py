import asyncio
import time

from version_1 import main as main_v1
from version_2 import load_all_data


async def bench_v1() -> float:
    start = time.perf_counter()
    result = main_v1()
    if asyncio.iscoroutine(result):
        await result
    else:
        await asyncio.to_thread(main_v1)
    return time.perf_counter() - start


async def bench_v2() -> float:
    start = time.perf_counter()
    await load_all_data("2025-07-08", "2025-07-10", "10:00:00", "18:45:00")
    return time.perf_counter() - start


async def main(n_runs: int = 5):
    print(f"Бенчмарк ({n_runs} прогонов):\n")
    for i in range(1, n_runs + 1):
        t1 = await bench_v1()
        t2 = await bench_v2()
        print(f"Run {i:>2}: v1 = {t1:.2f}s\t v2 = {t2:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
