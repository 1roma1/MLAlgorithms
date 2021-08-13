class Graph:
    __instance = None
    def __init__(self):
        self.operations = []
        self.constants = []
        self.variables = []

    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = Graph()
        return cls.__instance