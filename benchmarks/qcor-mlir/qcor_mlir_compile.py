import os, glob, time, csv
from datetime import datetime

dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirPath)

listOfSrcFiles = glob.glob(dirPath + "/qasm/*.qasm")

# Time to generate QIR (LLVM IR), and the total compile time
headers = ["Test Case", "qir-gen time", "llc time", "total time"]
firstWrite = True

# LL compiler
llc_exe = "~/.mlir/bin/llc"
result_file_name = 'result_mlir' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv'
# Compile with MLIR
for file in listOfSrcFiles:
  rowData = [os.path.splitext(os.path.basename(file))[0]]
  # Run MLIR tool
  start_time = time.time()
  os.system("qcor-mlir-tool " + file)
  rowData.append(time.time() - start_time)
  ll_file_name = dirPath + "/qasm/" + rowData[0] + ".ll"
  # Compile with LLC
  start_time = time.time()
  os.system(llc_exe + " " + ll_file_name)
  rowData.append(time.time() - start_time)

  # Run full QCOR compile (via MLIR)
  start_time = time.time()
  os.system("qcor " + file)
  rowData.append(time.time() - start_time)
  with open(result_file_name, 'a', newline='') as csvfile:
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
obj_files = glob.glob(dirPath + "/qasm/*.o")
for file_name in obj_files:
  os.remove(file_name)
s_files = glob.glob(dirPath + "/qasm/*.s")
for file_name in s_files:
  os.remove(file_name)