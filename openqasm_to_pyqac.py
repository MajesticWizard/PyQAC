import re

script = open("qasm_script.txt", "r")
new_script = open("pyqac_script.py", "w+")
new_script.write("import PyQAC.Quant as pq\n\n")
script.readline()
script.readline()
match = re.search('\d{1,2}', script.readline())
new_script.write(f"q = pq.QuantumVector({match[0]})\n")
for line in script:
    prefix = line.split()[0]
    postfix = line.split()[1:]
    match = re.search("\d{1,2}", postfix.pop())
    params_function = f"({match[0]}"
    if prefix in ["swap","cswap","srswap","csrswap","iswap", "xy"]:
        match = re.search("\d{1,2}", postfix.pop())
        params_function += f", {match[0]}"
    if not postfix:
        params_function += ')'
    else:
        params_function += ', '
        params_gate = []
        controling_bits = []
        for param in postfix[-1::-1]:
            match = re.search("\d{1,2}", param)
            if '(' in param and ')' in param:
                params_gate = match[0]
            elif "q" in param:
                controling_bits.append(int(match[0]))
            else:MajesticWizard/PyQAC
                params_gate.insert(0, int(match[0]))
        if controling_bits:
            params_function += str(controling_bits) + ", "
        if params_gate or params_gate == 0:
            params_function += str(params_gate) + ", "
        params_function = params_function[:-2:] + ')'
    if prefix == "ccx" or prefix == "cx":
        prefix = "cnot"
    new_script.write("q.add_" + prefix + params_function + "\n")
new_script.write("vector = q.result_vector() \n")
new_script.write("prob = q.probabilities() \n")
new_script.write("num = q.return_random_vector() \n")
new_script.write('res = {"res_vector": vector, "probabilities": prob, "res_num": num}')
