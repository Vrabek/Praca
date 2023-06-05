import os
import psutil
import time

from memory_profiler import memory_usage

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        result["time"] = elapsed_time
        
        return result
    return wrapper

def memory_tracker_basic(func):
    def wrapper(*args, **kwargs):
        proc = psutil.Process(os.getpid())
        mem_info1 = proc.memory_info()
        result = func(*args, **kwargs)
        mem_info2 = proc.memory_info()
        delta = mem_info2.rss - mem_info1.rss
        result["memory"] = delta / (1024 * 1024)
        return result
    return wrapper

def memory_tracker(func):
    def wrapper(*args, **kwargs):
        mem_usage_before = memory_usage(-1)[0]
        result = func(*args, **kwargs)
        mem_usage_after = memory_usage(-1)[0]
        mem_diff = mem_usage_after - mem_usage_before
        result["memory"] = mem_diff
        
        return result
    return wrapper


if __name__ == "__main__":

    @time_tracker
    @memory_tracker
    def example_function():
        result = {}
        result["output"] = sum([i for i in range(1000)])
        time.sleep(2) 
        return result

    result = example_function()
    print(result)