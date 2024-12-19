import functools
import json, os
from pathlib import Path


def persistent_cache(func):
    """
    Simple persistent cache decorator.
    Creates a "cache/" directory if it does not exist and writes the
    caches of the given func to the file "cache/<func-name>.cache"
    """
    file_path = Path(f'cache/{func.__name__}.cache')
    file_path.parent.mkdir(exist_ok=True)
    try:
        with open(file_path, 'r') as f:
            cache = json.load(f)
    except (IOError, ValueError):
        cache = {}

    @functools.wraps(func)
    def wrapper(*args, persistent_cache_key=None, **kwargs):
        """
            :param persistent_cache_key: The key to use for the cache. If None, the arguments of the function are used.
        """
        if persistent_cache_key:
            persistent_cache_key = str(persistent_cache_key)
        else:
            assert args or kwargs, f'Cannot create key without arguments or explicit key. Use persistent_cache_key=<key> or provide other arguments to {func.__name__}()'
            persistent_cache_key = str(args) + str(kwargs)

        if persistent_cache_key not in cache:
            cache[persistent_cache_key] = func(*args, **kwargs)
            with open(file_path, 'w') as f:
                json.dump(cache, f)
        return cache[persistent_cache_key]

    return wrapper
