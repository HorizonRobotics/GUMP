import gzip
import pickle
from typing import Any


def gload(filename: str) -> Any:
    """
    read compress pickle file
    """
    file = gzip.GzipFile(filename, 'rb')
    res = pickle.load(file)
    file.close()
    return res


def gdump(obj: Any, filename: str) -> None:
    """
    dump file into compressed pickle
    """
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file, -1)
    file.close()
