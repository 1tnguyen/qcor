import os, glob, time, csv
from datetime import datetime

dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirPath)

listOfSrcFiles = glob.glob(dirPath + "/qasm/*.qasm")

headers = ["Test Case", "Transpile Time"]
firstWrite = True
from qiskit import QuantumCircuit
from qiskit.compiler import transpile

for file in listOfSrcFiles:
  try: 
    rowData = [os.path.splitext(os.path.basename(file))[0]]
    # Time the transpile time
    start_time = time.time()
    qc = QuantumCircuit.from_qasm_file(file)
    result = transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'])
    rowData.append(time.time() - start_time)
    
    with open('result_qiskit' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv', 'a', newline='') as csvfile:
      resultWriter = csv.writer(csvfile)
      if firstWrite is True:
        resultWriter.writerow(headers)
        firstWrite = False
      # Write data
      resultWriter.writerow(rowData)
  except:
    print(file)