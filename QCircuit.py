from qiskit import QuantumCircuit
from qiskit import transpile
import numpy as np
from qiskit_aer import Aer
import torch
import torch.nn as nn




#================ qc setting ==================================
n_qubits = 4    # for 16 states
n_layers = 2
n_params = n_qubits * n_layers * 2  # 2 parameters per qubits
qc = QuantumCircuit(n_qubits)

#================ state trans =================================
# decimal -> binary
def dec_to_bin(state, n):
    return format(state, f'0{n}b')

#================ create qc ===================================
# encoding
def encoding_layer(state):
    for i in range(n_qubits):
        if dec_to_bin(state, n_qubits)[i] == '1':
            qc.rx(np.pi, i)  # apply RX gate with pi rotation
        else:
            qc.rx(0, i)  # apply RX gate with 0 rotation

# entanglement using CNOT gates
def entanglement_layer():
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

# Apply parameterized gates
def variational_layer(weights):
    for i in range(n_qubits):
        qc.ry(weights[i], i)
        qc.rz(weights[i], i)

# ==============================================================
def Q_value_list(iuput_state, weights, shots=1024):
    encoding_layer(iuput_state)
    
    for _ in range(n_layers):
        entanglement_layer()
        qc.barrier()
        variational_layer(weights)
        qc.barrier()
    
    qc.measure_all()
    
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(transpile(qc, backend), shots=shots)
    result = job.result()
    counts = result.get_counts()

    expectation_list = []
    for qubit in range(n_qubits):
        prob = 0
        for bitstring, count in counts.items():
            if bitstring[::-1][qubit] == '1':
                prob += count
        expectation = prob / shots
        expectation_list.append(expectation)
    
    return expectation_list


class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.randn(n_qubits * n_layers * 2))  # 2 parameters per qubit

    def forward(self, x):
        q_out = Q_value_list(x, self.weights.detach().numpy(), shots=1024)
        return torch.tensor(q_out, dtype=torch.float32)
    
# Example usage
vqc = VQC(n_qubits, n_layers)
input_state = 11
output = vqc(input_state)
print(f"Output for input state {input_state}: {output}")


