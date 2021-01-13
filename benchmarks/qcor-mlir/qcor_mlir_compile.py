import os, glob, time, csv

dirPath = os.path.dirname(os.path.realpath(__file__))

listOfSrcFiles = glob.glob(dirPath + "/qasm/*.qasm")

# Time to generate QIR (LLVM IR), and the total compile time
headers = ["Test Case", "qir-gen time", "total time"]
firstWrite = True

# Compile with MLIR
for file in listOfSrcFiles:
  rowData = [os.path.splitext(os.path.basename(file))[0]]
  # Run MLIR tool
  start_time = time.time()
  os.system("qcor-mlir-tool " + file)
  rowData.append(time.time() - start_time)
  # Run full QCOR compile (via MLIR)
  start_time = time.time()
  os.system("qcor " + file)
  rowData.append(time.time() - start_time)
  with open('result_mlir.csv', 'a', newline='') as csvfile:
    resultWriter = csv.writer(csvfile)
    if firstWrite is True:
      resultWriter.writerow(headers)
      firstWrite = False
    # Write data
    resultWriter.writerow(rowData)


# Clean-up any output files
ll_files = glob.glob(dirPath + "/qasm/*.ll")
for file_name in ll_files:
  os.remove(file_name)
bc_files = glob.glob(dirPath + "/qasm/*.bc")
for file_name in bc_files:
  os.remove(file_name)