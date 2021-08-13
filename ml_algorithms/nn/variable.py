from operation import Add, Multiply
from graph import Graph

class Placeholder:
    def __init__(self):
        self.consumers = []

        Graph.get_instance().constants.append(self)

class Variable:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []

        Graph.get_instance().variables.append(self)

    