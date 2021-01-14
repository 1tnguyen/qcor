import os, glob, time, csv
from datetime import datetime

dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirPath)

listOfSrcFiles = glob.glob(dirPath + "/qasm/*.qasm")
# We must run smaller files first.
# It may hang indefinitely for larger files...
listOfSrcFiles = sorted(listOfSrcFiles, key = os.path.getsize)

#Time to compile via syntax handler
headers = ["Test Case", "total time"]
firstWrite = True
result_file_name = 'result_csp' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv'
for file in listOfSrcFiles:
  print(file)
  rowData = [os.path.splitext(os.path.basename(file))[0]]
  start_time = time.time()
  os.system("qcor -c -DTEST_SOURCE_FILE=\\\"" + file + "\\\" qcor_csp.cpp")
  rowData.append(time.time() - start_time)
  with open(result_file_name, 'a', newline='') as csvfile:
    resultWriter = csv.writer(csvfile)
    if firstWrite is True:
      resultWriter.writerow(headers)
      firstWrite = False
    # Write data
    resultWriter.writerow(rowData)