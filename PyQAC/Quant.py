import random
import numpy as np
import PyQAC.MatrixGeneration as mg

class GateError(Exception):
    def __init__(self, text):
        self.text = text

def complex_modulus(z: complex): # finding the absolute value of complex number
    return np.sqrt(z.real ** 2 + z.imag ** 2)

class QuantumVector: # maximum number of qubits is 12
    def __init__(self, size):
        self.size = size
        self.gates = [[mg.I for i in range(size)]]
        self.vector = [1]
        self.vector.extend([0 for i in range(2**size - 1)])
        self.vector = np.matrix(self.vector)

    def result_matrix(self,position):
        res = 1
        for i in self.gates[position][-1::-1]:
            if type(i) is not int:
                res = np.kron(res,i)
        return res

    def final_gate(self):
        res = 1
        for i in range(len(self.gates)):
            res *= self.result_matrix(i)
        return res

    def result_vector(self):
        return (self.vector * self.final_gate()).tolist()[0]

    def probabilities(self, position=0, sz=-1):
        if sz <= 0: sz = self.size
        res = [0 for i in range(2**size)]#{i:0 for i in range(2**size)}
        vector = self.result_vector()
        for i in range(len(vector)):
            res[(i//2**position)%2**size] += complex_modulus(vector[i])**2
        return res

    def return_random_vector(self, position=0, sz=-1):
        if sz <= 0: sz = self.size
        probabilities = self.probabilities(position,sz)
        sum = 0
        rand = random.random()
        for i in range(len(probabilities)):
            sum += probabilities[i]
            if rand < sum:
                return i

    def add_h(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.H
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.H
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.H

    def add_x(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.X
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.X
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.X

    def add_y(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.Y
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.Y
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.Y

    def add_z(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.Z
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.Z
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.Z

    def add_srn(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.srn
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.srn
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.srn

    def add_srndg(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.srndg
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.srndg
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.srndg

    def add_r2(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.r2
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.r2
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.r2

    def add_r4(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.r4
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.r4
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.r4

    def add_r8(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.r8
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.r8
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.r8

    def add_sdg(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.sdg
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.sdg
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.sdg

    def add_tdg(self, position, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.tdg
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.tdg
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.tdg

    def add_rx(self, position, theta, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.rx(theta)
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.rx(theta)
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.rx(theta)

    def add_ry(self, position, theta, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.ry(theta)
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.ry(theta)
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.ry(theta)

    def add_rz(self, position, theta, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.rz(theta)
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.rz(theta)
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.rz(theta)

    def add_u1(self, position, theta, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.u1(theta)
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.u1(theta)
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.u1(theta)

    def add_u2(self, position, angles, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.u2(angles[0], angles[1])
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.u2(angles[0], angles[1])
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.u2(angles[0], angles[1])

    def add_u3(self, position, angles, ops=-1):
        if ops >= 0:
            if np.all(self.gates[ops][position] == mg.I):
                self.gates[ops][position] = mg.u3(angles[0], angles[1], angles[2])
            else:
                raise GateError('GateError: Gate is already occupied')
        else:
            for i in range(len(self.gates)):
                if np.array_equal(self.gates[i][position], mg.I):
                    self.gates[i][position] = mg.u3(angles[0], angles[1], angles[2])
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][position] = mg.u3(angles[0], angles[1], angles[2])


    def add_swap(self, first_bit, second_bit, ops=-1):
        bottom_bound = min(first_bit, second_bit)
        upper_bound = max(first_bit, second_bit)
        inp1 = 0
        inp2 = upper_bound
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_swap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_swap(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_swap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_iswap(self, first_bit, second_bit, ops=-1):
        bottom_bound = min(first_bit, second_bit)
        upper_bound = max(first_bit, second_bit)
        inp1 = 0
        inp2 = upper_bound
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_iswap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_iswap(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_iswap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_srswap(self, first_bit, second_bit, ops=-1):
        bottom_bound = min(first_bit, second_bit)
        upper_bound = max(first_bit, second_bit)
        inp1 = 0
        inp2 = upper_bound
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_srswap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_srswap(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_srswap(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_xy(self, first_bit, second_bit, phi, ops=-1):
        bottom_bound = min(first_bit, second_bit)
        upper_bound = max(first_bit, second_bit)
        inp1 = 0
        inp2 = upper_bound
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_XY(inp1, inp2, phi)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_XY(inp1, inp2, phi)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_XY(inp1, inp2, phi)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cnot(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cnot(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cnot(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cnot(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cy(self, controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cy(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cy(inp1, inp2)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cy(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cz(self, controled_bit, control_bits, ops = -1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cz(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cz(inp1, inp2)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cz(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_ch(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_ch(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_ch(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_ch(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_csrn(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_csrn(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_csrn(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_csrn(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cr2(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cr2(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cr2(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cr2(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cr4(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cr4(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cr4(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cr4(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cr8(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cr8(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cr8(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cr8(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_crx(self,controled_bit, control_bits, theta, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_crx(inp1, inp2, theta)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_crx(inp1, inp2, theta)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_crx(inp1, inp2, theta)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cry(self, controled_bit, control_bits, theta, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cry(inp1, inp2, theta)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cry(inp1, inp2, theta)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cry(inp1, inp2, theta)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_crz(self, controled_bit, control_bits, phi, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_crz(inp1, inp2, phi)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_crz(inp1, inp2, phi)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_crz(inp1, inp2, phi)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cu1(self, controled_bit, control_bits, lmbd, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cu1(inp1, inp2, lmbd)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cu1(inp1, inp2, lmbd)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cu1(inp1, inp2, lmbd)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cu2(self, controled_bit, control_bits, angles, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cu2(inp1, inp2, angles)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cu2(inp1, inp2, angles)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cu2(inp1, inp2, angles)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cu3(self, controled_bit, control_bits, angles, ops=-1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cu3(inp1, inp2, angles)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound + 1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cu3(inp1, inp2, angles)
                    for j in range(bottom_bound + 1, upper_bound + 1):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cu3(inp1, inp2, angles)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_csdg(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_csdg(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_csdg(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_csdg(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_ctdg(self,controled_bit, control_bits, ops=-1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_ctdg(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_ctdg(inp1, inp2)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_ctdg(inp1, inp2)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_cswap(self, first_bit, second_bit, control_bits, ops=-1):
        bottom_bound = min(first_bit, second_bit,min(control_bits))
        upper_bound = max(first_bit, second_bit,max(control_bits))
        if first_bit < second_bit:
            inp1 = first_bit - bottom_bound
            inp2 = second_bit - bottom_bound
        else:
            inp1 = second_bit - bottom_bound
            inp2 = first_bit - bottom_bound
        inp3 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_cswap(inp1, inp2, inp3)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_cswap(inp1, inp2, inp3)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_cswap(inp1, inp2, inp3)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_csrswap(self, first_bit, second_bit, control_bits, ops=-1):
        bottom_bound = min(first_bit, second_bit,min(control_bits))
        upper_bound = max(first_bit, second_bit,max(control_bits))
        if first_bit < second_bit:
            inp1 = first_bit - bottom_bound
            inp2 = second_bit - bottom_bound
        else:
            inp1 = second_bit - bottom_bound
            inp2 = first_bit - bottom_bound
        inp3 = [control_bit - bottom_bound for control_bit in control_bits]
        if ops >= 0:
            for j in range(bottom_bound, upper_bound + 1):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][bottom_bound] = mg.generate_csrswap(inp1, inp2, inp3)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(bottom_bound, upper_bound+1):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][bottom_bound] = mg.generate_csrswap(inp1, inp2, inp3)
                    for j in range(bottom_bound+1, upper_bound+1):
                     self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][bottom_bound] = mg.generate_csrswap(inp1, inp2, inp3)
            for j in range(bottom_bound + 1, upper_bound + 1):
                self.gates[len(self.gates) - 1][j] = 0

    def add_QFT(self, starting_pos, size, ops = -1):
        if ops >= 0:
            for j in range(starting_pos, starting_pos + size):
                if not np.array_equal(self.gates[ops][j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][starting_pos] = mg.generate_QFT(size)
            for j in range(starting_pos + 1, starting_pos + size):
                self.gates[ops][j] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(starting_pos,starting_pos + size):
                    if not np.array_equal(self.gates[i][j], mg.I):
                        break
                else:
                    self.gates[i][starting_pos] = mg.generate_QFT(size)
                    for j in range(starting_pos + 1, starting_pos + size):
                        self.gates[i][j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates) - 1][starting_pos] = mg.generate_QFT(size)
            for j in range(starting_pos + 1, starting_pos + size):
                self.gates[len(self.gates) - 1][j] = 0

    def add_power(self, starting_pos, n, B, R, ops = -1):
        m = 0
        while 2 ** m < R:
            m += 1
        if ops >=0:
            for j in range(n + m):
                if not np.array_equal(self.gates[ops][starting_pos + j], mg.I):
                    raise GateError('GateError: Gate is already occupied')
            self.gates[ops][starting_pos] = mg.generate_pow_matrix(n, B, R)
            for i in range(starting_pos + 1, starting_pos + n + m):
                self.gates[ops][i] = 0
        else:
            for i in range(len(self.gates)):
                for j in range(n + m):
                    if not np.array_equal(self.gates[i][starting_pos + j], mg.I):
                        break
                else:
                    self.gates[i][starting_pos] = mg.generate_pow_matrix(n, B, R)
                    for j in range(starting_pos + 1, starting_pos + n + m):
                        self.gates[j] = 0
                    return None
            self.gates.append([mg.I for i in range(self.size)])
            self.gates[len(self.gates)-1][starting_pos] = mg.generate_pow_matrix(n, B, R)
            for i in range(starting_pos + 1,starting_pos + n+m):
                self.gates[len(self.gates)-1][i] = 0


if __name__ == '__main__':
    q = QuantumVector(2)
    q.add_h(0)
    q.add_h(1)
    print(q.ret)
    #обычный алгоритм Шора
   # q1 = QuantumVector(12)
  #  for i in range(6):
  #      q1.add_h(i)
  #  q1.add_power(0, 6, 7, 36)
  #  q1.add_QFT(0, 6)
  #  answer = q1.return_random_vector(0, 6)
