import numpy as np

Zero = np.matrix([[0, 0],
                  [0, 0]]) #* complex(1)
I = np.matrix(([[1, 0],
                [0, 1]])) * complex(1)
X = np.matrix([[0, 1],
               [1, 0]])#* complex(1)
H = np.matrix([[1, 1],
               [1, -1]]) * complex(1/np.sqrt(2))
Y = np.matrix([[complex(0), complex(0,-1)],
               [complex(0,1), complex(0)]])
Z = np.matrix([[1, 0],
               [0, -1]]) * complex(1)
srn = np.matrix([[complex(0.5, 0.5), complex(0.5, -0.5)],
                 [complex(0.5, -0.5), complex(0.5, 0.5)]])
srndg = np.matrix([[complex(0.5, -0.5), complex(0.5, 0.5)],
                   [complex(0.5, 0.5), complex(0.5, -0.5)]])
r2 = np.matrix([[1, 0],
                [0, np.e ** complex(0, np.pi / 2)]])
r4 = np.matrix([[1, 0],
                [0, np.e ** complex(0, np.pi / 4)]])
r8 = np.matrix([[1, 0],
                [0, np.e ** complex(0, np.pi / 8)]])
sdg = np.matrix([[1, 0],
                [0, np.e ** complex(0, -np.pi / 2)]])
tdg = np.matrix([[1, 0],
                [0, np.e ** complex(0, -np.pi / 4)]])


def rx(theta):
    matrix = np.matrix([[np.cos(theta / 2), complex(0, -np.sin(theta / 2))],
                        [complex(0, -np.sin(theta / 2)), np.cos(theta / 2)]])
    return matrix


def ry(theta):
    matrix = np.matrix([[np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)]])
    return matrix


def rz(phi):
    matrix = np.matrix([[complex(np.cos(phi / 2), -np.sin(phi / 2)), 0],
                        [0, complex(np.cos(phi / 2), np.sin(phi / 2))]])
    return matrix


def u1(theta):
    matrix = np.matrix([[1, 0],
                        [0, np.e ** complex(0, theta)]])
    return matrix


def u2(theta, phi):
    matrix = np.matrix([[1/np.sqrt(2), -(np.e ** complex(0, theta)) / np.sqrt(2)],
                        [(np.e ** complex(0, phi)) / np.sqrt(2), (np.e ** complex(0, theta + phi)) / np.sqrt(2)]])
    return matrix


def u3(theta, phi, lmbda):
    matrix = np.matrix([[np.cos(theta/2), -(np.e ** complex(0, lmbda)) * np.sin(theta / 2)],
                        [(np.e ** complex(0, phi)) * np.sin(theta / 2), (np.e ** complex(0, lmbda + phi)) * np.cos(theta / 2)]])
    return matrix

def n_root_of_1(n):
    return complex(np.cos(np.pi / (2**(n-1))), np.sin(np.pi / (2 ** (n - 1))))


def generate_cnot(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits))
    result_matrix = I
    useless_bit = []
    start_of_permutation = 0
    for i in control_bits:
        start_of_permutation += 2 ** i

    for i in range(size):
        if i not in control_bits and i != contoled_bit:
            useless_bit.append(i)

    for i in range(size):
        result_matrix = np.kron(I, result_matrix)
    if useless_bit != []:
        for i in range(2 ** len(useless_bit)):
            temp = [*map(int, bin(i).replace('0b', ''))]

            permutation = start_of_permutation
            for j in range(len(temp)):
                result_matrix[permutation], result_matrix[permutation + 2 ** contoled_bit] = result_matrix[
                                                                                                 permutation + 2 ** contoled_bit], \
                                                                                             result_matrix[
                                                                                                 permutation].copy()
    else:
        result_matrix[start_of_permutation], result_matrix[start_of_permutation + 2 ** contoled_bit] = result_matrix[
                                                                                                           start_of_permutation + 2 ** contoled_bit], \
                                                                                                       result_matrix[
                                                                                                           start_of_permutation].copy()

    return result_matrix


def generate_swap(first_q, second_q):
    distance = second_q - first_q + 1
    not_np_matrix = []
    for i in range(2**distance):
        bin_num = format(i, 'b')
        for j in range(distance - len(bin_num)):
            bin_num = '0' + bin_num
        bin_num = bin_num[distance-1] + bin_num[1:distance-1] + bin_num[0]
        lst = [0 for i in range(2**distance)]
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_srswap(first_q, second_q):
    distance = second_q - first_q + 1
    not_np_matrix = []
    for i in range(2**distance):
        bin_num1 = format(i, 'b')
        for j in range(distance - len(bin_num1)):
            bin_num1 = '0' + bin_num1
        bin_num2 = bin_num1[distance-1] + bin_num1[1:distance-1] + bin_num1[0]
        lst = [0 for i in range(2**distance)]
        if bin_num1 != bin_num2:
            lst[int(bin_num1, 2)] = complex(0.5, 0.5)
            lst[int(bin_num2, 2)] = complex(0.5, -0.5)
        else:
            lst[int(bin_num1, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_iswap(first_q, second_q):
    distance = second_q - first_q + 1
    not_np_matrix = []
    for i in range(2**distance):
        bin_num1 = format(i, 'b')
        for j in range(distance - len(bin_num1)):
            bin_num1 = '0' + bin_num1
        bin_num2 = bin_num1[distance-1] + bin_num1[1:distance-1] + bin_num1[0]
        lst = [0 for i in range(2**distance)]
        if bin_num1 != bin_num2:
            lst[int(bin_num2, 2)] = complex(0, 1)
        else:
            lst[int(bin_num1, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_XY(first_q, second_q, phi):
    distance = second_q - first_q + 1
    not_np_matrix = []
    for i in range(2**distance):
        bin_num1 = format(i, 'b')
        for j in range(distance - len(bin_num1)):
            bin_num1 = '0' + bin_num1
        bin_num2 = bin_num1[distance-1] + bin_num1[1:distance-1] + bin_num1[0]
        lst = [0 for i in range(2**distance)]
        if bin_num1 != bin_num2:
            lst[int(bin_num1, 2)] = np.cos(phi/2)
            lst[int(bin_num2, 2)] = complex(0, np.sin(phi/2))
        else:
            lst[int(bin_num1, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_ch(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1 / np.sqrt(2)
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = 1/np.sqrt(2)
            else:
                lst[int(bin_num, 2)] = -1 / np.sqrt(2)
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = 1 / np.sqrt(2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_csrn(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = complex(0.5, 0.5)
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0.5, -0.5)
            else:
                lst[int(bin_num, 2)] = complex(0.5, 0.5)
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] =  complex(0.5, -0.5)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_cr2(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = 1/np.sqrt(2)
            else:
                lst[int(bin_num, 2)] = np.e ** complex(0, np.pi / 2)
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = 1 / np.sqrt(2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_cr4(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = 1/np.sqrt(2)
            else:
                lst[int(bin_num, 2)] = np.e ** complex(0, np.pi / 4)
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = 1 / np.sqrt(2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_cr8(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = 1/np.sqrt(2)
            else:
                lst[int(bin_num, 2)] = np.e ** complex(0, np.pi / 8)
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = 1 / np.sqrt(2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_crx(contoled_bit, control_bits, theta):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = np.cos(theta / 2)
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0, -np.sin(theta/2))
            else:
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = complex(0,-np.sin(theta/2))
                lst[int(bin_num, 2)] = np.cos(theta / 2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_cry(contoled_bit, control_bits, theta):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = np.cos(theta / 2)
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = np.sin(theta/2)
            else:
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = -np.sin(theta/2)
                lst[int(bin_num, 2)] = np.cos(theta / 2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)

    return np.matrix(not_np_matrix).T


def generate_crz(contoled_bit, control_bits, phi):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = complex(np.cos(phi / 2), -np.sin(phi/2))
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0, -np.sin(theta/2))
            else:
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = complex(0,-np.sin(theta/2))
                lst[int(bin_num, 2)] = complex(np.cos(phi / 2), np.sin(phi/2))
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_cu1(contoled_bit, control_bits, theta):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0, -np.sin(theta/2))
            else:
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = complex(0,-np.sin(theta/2))
                lst[int(bin_num, 2)] = np.e ** complex(0,theta)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_cu2(contoled_bit, control_bits, angles):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1 / np.sqrt(2)
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = np.e ** complex(0, angles[0]) \
                                                                                       / np.sqrt(2)
            else:
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = -np.e ** complex(0, angles[1])\
                                                                                       / np.sqrt(2)
                lst[int(bin_num, 2)] = np.e ** complex(0, angles[0] + angles[1]) / np.sqrt(2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_cu3(contoled_bit, control_bits, angles):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = np.cos(angles[0])
                lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = np.e ** complex(0, angles[1]) \
                                                                                       * np.sin(angles[0] / 2)
            else:
                lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = -np.e ** complex(0, angles[2])\
                                                                                       * np.sin(angles[0] / 2)
                lst[int(bin_num, 2)] = np.e ** complex(0, angles[2] + angles[1]) * np.cos(angles[0] / 2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_csdg(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0, -np.sin(theta/2))
            else:
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = complex(0,-np.sin(theta/2))
                lst[int(bin_num, 2)] = np.e ** complex(0,-np.pi/2)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_ctdg(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[control_bit] == "0":
                break
        else:
            if bin_num[contoled_bit] == '0':
                lst[int(bin_num, 2)] = 1
                #lst[int(bin_num[:contoled_bit] + '1' + bin_num[contoled_bit+1:], 2)] = complex(0, -np.sin(theta/2))
            else:
                #lst[int(bin_num[:contoled_bit] + '0' + bin_num[contoled_bit + 1:], 2)] = complex(0,-np.sin(theta/2))
                lst[int(bin_num, 2)] = np.e ** complex(0,-np.pi/4)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_cy(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits))
    result_matrix = I
    useless_bit = []
    start_of_permutation = 0
    for i in control_bits:
        start_of_permutation += 2 ** i

    for i in range(size):
        if i not in control_bits and i != contoled_bit:
            useless_bit.append(i)

    for i in range(size):
        result_matrix = np.kron(I, result_matrix)
    if useless_bit != []:
        for i in range(2 ** len(useless_bit)):
            temp = [*map(int, bin(i).replace('0b', ''))]
            permutation = start_of_permutation
            for j in range(len(temp)):
                permutation += temp[j] * (2 ** useless_bit[j])
            result_matrix[permutation], result_matrix[permutation + 2 ** contoled_bit] = result_matrix[
                                                                                             permutation + 2 ** contoled_bit] * complex(
                0, 1), result_matrix[permutation].copy() * complex(0, -1)
    else:
        result_matrix[start_of_permutation], result_matrix[start_of_permutation + 2 ** contoled_bit] = result_matrix[
                                                                                                           start_of_permutation + 2 ** contoled_bit] * complex(
            0, 1), result_matrix[start_of_permutation].copy() * complex(0, -1)

    return result_matrix




def generate_cz(contoled_bit, control_bits):
    size = max(contoled_bit, max(control_bits))
    result_matrix = I
    useless_bit = []
    start_of_permutation = 0
    for i in control_bits:
        start_of_permutation += 2 ** i

    for i in range(size):
        if i not in control_bits and i != contoled_bit:
            useless_bit.append(i)
    for i in range(size):
        result_matrix = np.kron(I, result_matrix)

    if useless_bit != []:
        for i in range(2 ** len(useless_bit)):
            temp = [*map(int, bin(i).replace('0b', ''))]
            permutation = start_of_permutation

            for j in range(len(temp)):
                permutation += temp[j] * (2 ** useless_bit[j])

            result_matrix[permutation + 2 ** contoled_bit] = result_matrix[permutation + 2 ** contoled_bit] * -1
    else:
        result_matrix[start_of_permutation + 2 ** contoled_bit] = result_matrix[
                                                                      start_of_permutation + 2 ** contoled_bit] * -1


    return result_matrix


def generate_cswap(bit_swap1, bit_swap2, control_bits):
    size = max(bit_swap1, bit_swap2, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[bit_swap1] == bin_num[bit_swap2]:
                break
            if bin_num[control_bit] == "0":
                break
        else:
            bin_num = bin_num[:bit_swap1] + bin_num[bit_swap2] + bin_num[bit_swap1+1:bit_swap2] + bin_num[bit_swap1] + \
                      bin_num[bit_swap2+1::]
            lst[int(bin_num, 2)] = 1
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_csrswap(bit_swap1, bit_swap2, control_bits):
    size = max(bit_swap1, bit_swap2, max(control_bits)) + 1
    not_np_matrix = []
    for i in range(2**size):
        bin_num = format(i, 'b')
        for j in range(size - len(bin_num)):
            bin_num = '0' + bin_num
        lst = [0 for i in range(2**size)]
        for control_bit in control_bits:
            if bin_num[bit_swap1] == bin_num[bit_swap2]:
                break
            if bin_num[control_bit] == "0":
                break
        else:
            lst[int(bin_num,2)] = complex(0.5,0.5)
            bin_num = bin_num[:bit_swap1] + bin_num[bit_swap2] + bin_num[bit_swap1+1:bit_swap2] + bin_num[bit_swap1] + \
                      bin_num[bit_swap2+1::]
            lst[int(bin_num, 2)] = complex(0.5,-0.5)
            not_np_matrix.append(lst)
            continue
        lst[int(bin_num, 2)] = 1
        not_np_matrix.append(lst)
    return np.matrix(not_np_matrix).T


def generate_QFT(n):
    N = 2**n
    w = n_root_of_1(n)
    matrix = [[w**(i*j) for j in range(N)] for i in range(N)]
    return np.matrix(matrix) * 1/np.sqrt(N)


def generate_pow_matrix(n,B,R): #B**A mod R, n is number of qubits used as input
    m = 0
    matrix = []
    while 2**m < R:
        m+=1
    for i in range(2**(n+m)):
        x = i%(2**m)
        x_bin = [*map(int, bin(x).replace('0b', ''))]
        y = i // (2 ** m)
        y_bin = [*map(int, bin(y).replace('0b', ''))]
        fx = (B ** x) % R
        fx_bin = [*map(int, bin(fx).replace('0b', ''))]

        while len(x_bin) < n:
            x_bin.insert(0,0)
        while len(y_bin) < m:
            y_bin.insert(0,0)
        while len(fx_bin) < m:
            fx_bin.insert(0,0)

        for i in range(m):
            fx_bin[i] = (fx_bin[i] + y_bin[i]) % 2
        fx_bin.extend(x_bin)
        one_pos = 0
        for i in range(len(fx_bin)):
            one_pos += (2**i) * fx_bin[m+n-1-i]
        string = [0 for i in range(2**(n+m))]
        string[one_pos] = 1
        matrix.append(string)
    return np.matrix(matrix)


if __name__ == "__main__":
    print(generate_swap(1,4))
    #print(generate_ch(1,[0,2,3,4,5,6,7,8,9,10,11]))
    #print(generate_cy(1,[0,2,3,4,5,6,7,8,9,10,11]))