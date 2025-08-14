from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn




#================ qc setting ==================================
n_qubits = 4    # for 16 states
n_layers = 1
n_params = n_qubits * n_layers * 2  # 2 parameters per qubits
shots = 1024
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
        qc.rz(weights[i + n_qubits], i)

# ==============================================================
def Q_value_list(iuput_state, weights, shots=shots):
    encoding_layer(iuput_state)
    
    for _ in range(n_layers):
        qc.barrier()
        entanglement_layer()
        qc.barrier()
        variational_layer(weights)
    
    qc.measure_all()
    
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(transpile(qc, backend), shots=shots)
    result = job.result()
    counts = result.get_counts()
    # plot_histogram(counts)

    expectation_list = []
    for qubit in range(n_qubits):
        prob = 0
        for bitstring, count in counts.items():
            print(f"Bitstring: {bitstring}, Count: {count}")
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

    def forward(self, x, weights):
        q_out = Q_value_list(x, weights.detach().numpy(), shots=shots)
        return torch.tensor(q_out, dtype=torch.float32)
    
# Example usage
weights = torch.tensor(np.random.rand(n_params), dtype=torch.float32, requires_grad=True)
vqc = VQC(n_qubits, n_layers)
input_state = 11 
output = vqc(input_state, weights)

print(f"Output for input state {input_state}: {output}")

# Plot
fig = qc.draw(output='mpl')
plt.title(f"Quantum Circuit with n_layers={n_layers}")
# plt.savefig("QC_1layer.png", dpi=300, bbox_inches='tight')
plt.show()


