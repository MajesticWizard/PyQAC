import random
import numpy as np
import MatrixGeneration as mg
from math import gcd

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
        return self.vector * self.final_gate()

    def probabilities(self, position, size):
        res = [0 for i in range(2**size)]#{i:0 for i in range(2**size)}
        vector = self.result_vector().tolist()
        for i in range(len(vector[0])):
            res[(i//2**position)%2**size] += complex_modulus(vector[0][i])**2
        return res

    def return_random_vector(self, position, size):
        probabilities = self.probabilities(position,size)
        sum = 0
        rand = random.random()
        for i in range(len(probabilities)):
            sum += probabilities[i]
            if rand < sum:
                return i


    def add_H(self, position, ops = -1):
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

    def add_X(self, position, ops =-1):
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

    def add_Y(self, position, ops = -1):
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

    def add_Z(self, position, ops = -1):
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

    def add_cnot(self,controled_bit, control_bits, ops = -1):
        bottom_bound = min(controled_bit,min(control_bits))
        upper_bound = max(controled_bit,max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = []
        for i in control_bits:
            inp2.append(i - bottom_bound)
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

    def add_cy(self, controled_bit, control_bits, ops = -1):
        bottom_bound = min(controled_bit, min(control_bits))
        upper_bound = max(controled_bit, max(control_bits))
        inp1 = controled_bit - bottom_bound
        inp2 = []
        for i in control_bits:
            inp2.append(i - bottom_bound)
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
        inp2 = []
        for i in control_bits:
            inp2.append(i - bottom_bound)
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
    #обычный алгоритм Шора
    q1 = QuantumVector(12)
    for i in range(6):
        q1.add_H(i)
    q1.add_power(0, 6, 7, 36)
    q1.add_QFT(0, 6)
    answer = q1.return_random_vector(0, 6)
