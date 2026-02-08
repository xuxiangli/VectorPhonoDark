from vectorphonodark.physics import get_q_max


input_path = "/Users/jukcoeng/Desktop/Dark_Matter/Vector Space Integration/VectorPhonoDark/"
material_input= input_path + "inputs/material/GaAs/GaAs.py"

factor = 10.0

q_max = get_q_max(material_input=material_input, factor=factor)

print(f"q_max = {q_max} eV")