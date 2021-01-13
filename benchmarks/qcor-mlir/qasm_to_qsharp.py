import os, glob
# Staq tool qsharp_compiler tool
qasm_to_qsharp = "staq/build/tools/qsharp_compiler"


dirPath = os.path.dirname(os.path.realpath(__file__))
listOfSrcFiles = glob.glob(dirPath + "/qasm/*.qasm")

# Compile QASM to QSharp files
for file in listOfSrcFiles:
  result_qs = dirPath + "/qsharp/" + os.path.splitext(os.path.basename(file))[0] + ".qs"
  os.system(qasm_to_qsharp + " < " + file + " > " + result_qs)