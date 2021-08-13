from variable import Variable
from operation import Operation
from gradients import _gradinet_registry

from queue import Queue

def compute_gradients(loss):
    grad_table = {}

    grad_table[loss] = 1

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        if node != loss:
            grad_table[node] = 0

            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]

                consumer_op_type = consumer.__class__
                bprop = _gradinet_registry[consumer_op_type]

                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    lossgrads_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrads_wrt_node

        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)
    return grad_table