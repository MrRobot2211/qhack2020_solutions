#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    from pennylane import qaoa

    dev = qml.device("default.qubit", wires=NODES)#, analytic=True, shots=2)
    
    #pauli_z = [[1, 0], [0, -1]]
    #pauli_z_6 =    np.kron(pauli_z, np.kron(pauli_z,np.kron(pauli_z,np.kron(pauli_z,np.kron(pauli_z, pauli_z)))))
    
    U_C, U_B = qaoa.max_independent_set(graph,constrained=True)
    
    def comp_basis_measurement(wires):
        n_wires = len(wires)
        return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, U_C)
        qaoa.mixer_layer(alpha, U_B)

    @qml.qnode(dev)
    def circuit( params):
        # apply Hadamards to get the n qubit |+> state
#         for wire in range(n_wires):
#             qml.Hadamard(wires=wire)
        # p instances of unitary operators
        gammas = params[0]
        alphas = params[1]
        
        qml.layer(qaoa_layer, N_LAYERS, gammas, alphas)
       
        return  qml.probs(range(NODES)) #qml.sample(qml.PauliZ(0))#qml.probs(wires=[i for i in range(NODES)])
    
    max_base_vec = np.argmax(circuit(params))
    
    bin_rep = np.binary_repr(max_base_vec, width=NODES)

    for i in range(NODES):
        if bin_rep[i] =='1':
            max_ind_set.append(i)

        # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
