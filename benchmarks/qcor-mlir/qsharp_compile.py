# NOTE: we **only** compile the Q# source using the
# Q# command line tool, i.e. not building the whole .NET project.
# We don't time the QIR generation, just using the **Release** version
# of the QSC (Q# compiler) to compile the Q# source file.
import os, glob, time, csv
from datetime import datetime

dirPath = os.path.dirname(os.path.realpath(__file__))
os.chdir(dirPath + "/qsharp")

listOfSrcFiles = glob.glob(dirPath + "/qsharp/*.qs")
# We must run smaller files first.
# It may hang indefinitely for larger files...
listOfSrcFiles = sorted(listOfSrcFiles, key = os.path.getsize)

#Time to compile via syntax handler
headers = ["Test Case", "total time"]
firstWrite = True
# HARD-CODED qsc location, updated according to your system
qsc_exe = "/root/.nuget/packages/microsoft.quantum.sdk/0.14.2011120240/tools/qsc/qsc.dll"
# Q# build config file (has a placeholder for the qs source file)
response_file = dirPath + "/qsharp/qsc.rsp"
result_file_name = 'result_qsharp' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv'
for file in listOfSrcFiles:
  rowData = [os.path.splitext(os.path.basename(file))[0]]
  with open(response_file) as f:
    # Replace the particular file name
    newText=f.read().replace("$$FILE_NAME$$", os.path.basename(file))
  with open(dirPath + "/qsharp/qsc_run.rsp", "w") as f:
    f.write(newText)
  
  # Start compilation
  start_time = time.time()
  os.system("dotnet \"" + qsc_exe + "\" build --response-files qsc_run.rsp")
  rowData.append(time.time() - start_time)
  with open(, 'a', newline='') as csvfile:
    resultWriter = csv.writer(csvfile)
    if firstWrite is True:
      resultWriter.writerow(headers)
      firstWrite = False
    # Write data
    resultWriter.writerow(rowData)