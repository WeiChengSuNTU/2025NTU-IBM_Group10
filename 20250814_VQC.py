from qiskit import QuantumCircuit
import numpy as np


#================ qc setting ==================================
n_qubits = 4
qc = QuantumCircuit(n_qubits)
weights = [0.1, 0.2, 0.3, 0.4]  
#================ state trans =================================
state = 11
def dec_to_bin(state, n):
    return format(state, f'0{n}b')

#================ create qc ===================================
# Encode the state into the quantum circuit
for i in range(n_qubits):
    state_bit = dec_to_bin(state, n_qubits)[i]
    if state_bit == '1':
        qc.rx(np.pi, i)     # encode state with RX gates
    else:
        qc.rx(0, i)         # RX gate with angle 0 for state 0
qc.barrier()

# entanglement using CNOT gates
def entanglement_layer():
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

# Apply parameterized gates
def variational_layer(weights):
    for i in range(n_qubits):
        qc.ry(weights[i], i)
        qc.rz(weights[i], i)

# repeat the ansatz layer
for _ in range(2):
    entanglement_layer()
    qc.barrier()
    variational_layer(weights)
    qc.barrier()


#================ measure qc ==================================
qc.measure_all() 

print(qc.draw())    # visualize the circuit