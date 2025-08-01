import sys

class Logger:
    """
    A custom logger class that writes output to both the console and a log file.
    This is a singleton to ensure only one instance is created.
    """
    _instance = None

    def __new__(cls, filename="log.txt"):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.terminal = sys.stdout
            cls._instance.log = open(filename, "a", encoding='utf-8')
            print("Logger initialized")
        return cls._instance

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
