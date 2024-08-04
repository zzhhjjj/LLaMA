import sys
from functools import wraps

def redirect_output(file_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with open(file_path, 'w') as f:
                # Redirect stdout and stderr to the file
                sys.stdout = f
                sys.stderr = f
                try:
                    return func(*args, **kwargs)
                finally:
                    # Reset stdout and stderr to default values
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
        return wrapper
    return decorator

class RedirectOutput:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        if self.file_path:
            self.f = open(self.file_path, 'w')
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = self.f
            sys.stderr = self.f

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_path:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            self.f.close()