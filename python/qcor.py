import xacc
from _pyqcor import *
import sys
import inspect
from typing import List
import typing
import re
from collections import defaultdict

List = typing.List
PauliOperator = xacc.quantum.PauliOperator


def X(idx):
    return xacc.quantum.PauliOperator({idx: 'X'}, 1.0)


def Y(idx):
    return xacc.quantum.PauliOperator({idx: 'Y'}, 1.0)


def Z(idx):
    return xacc.quantum.PauliOperator({idx: 'Z'}, 1.0)


# Simple graph class to help resolve kernel dependency (via topological sort)
class KernelGraph(object):
    def __init__(self):
        self.graph = defaultdict(list)
        self.V = 0
        self.kernel_idx_dep_map = {}
        self.kernel_name_list = []

    def addKernelDependency(self, kernelName, depList):
        self.kernel_name_list.append(kernelName)
        self.kernel_idx_dep_map[self.V] = []
        for dep_ker_name in depList:
            self.kernel_idx_dep_map[self.V].append(
                self.kernel_name_list.index(dep_ker_name))
        self.V += 1

    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Topological Sort.
    def topologicalSort(self):
        self.graph = defaultdict(list)
        for sub_ker_idx in self.kernel_idx_dep_map:
            for dep_sub_idx in self.kernel_idx_dep_map[sub_ker_idx]:
                self.addEdge(dep_sub_idx, sub_ker_idx)

        in_degree = [0]*(self.V)
        for i in self.graph:
            for j in self.graph[i]:
                in_degree[j] += 1

        queue = []
        for i in range(self.V):
            if in_degree[i] == 0:
                queue.append(i)
        cnt = 0
        top_order = []
        while queue:
            u = queue.pop(0)
            top_order.append(u)
            for i in self.graph[u]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
            cnt += 1

        sortedDep = []
        for sorted_dep_idx in top_order:
            sortedDep.append(self.kernel_name_list[sorted_dep_idx])
        return sortedDep

    def getSortedDependency(self, kernelName):
        kernel_idx = self.kernel_name_list.index(kernelName)
        # No dependency
        if len(self.kernel_idx_dep_map[kernel_idx]) == 0:
            return []

        sorted_dep = self.topologicalSort()
        result_dep = []
        for dep_name in sorted_dep:
            if dep_name == kernelName:
                return result_dep
            else:
                result_dep.append(dep_name)


class qjit(object):
    """
    The qjit class serves a python function decorator that enables 
    the just-in-time compilation of quantum python functions (kernels) using 
    the QCOR QJIT infrastructure. Example usage:

    @qjit
    def kernel(qbits : qreg, theta : float):
        X(q[0])
        Ry(q[1], theta)
        CX(q[1], q[0])
        for i in range(q.size()):
            Measure(q[i])

    q = qalloc(2)
    kernel(q)
    print(q.counts())

    Upon initialization, the python inspect module is used to extract the function body 
    as a string. This string is processed to create a corresponding C++ function with 
    pythonic function body as an embedded domain specific language. The QCOR QJIT engine 
    takes this function string, and delegates to the QCOR Clang SyntaxHandler infrastructure, which 
    maps this function to a QCOR QuantumKernel sub-type, compiles to LLVM bitcode, caches that bitcode 
    for future fast lookup, and extracts function pointers using the LLVM JIT engine that can be called 
    later, affecting execution of the quantum code. 

    Note that kernel function arguments must provide type hints, and allowed types are int, bool, float, List[float], and qreg. 

    qjit annotated functions can also be passed as general functors to other QCOR API calls like 
    createObjectiveFunction, and createModel from the QSim library. 

    """

    def __init__(self, function, *args, **kwargs):
        """Constructor for qjit, takes as input the annotated python function and any additional optional
        arguments that are used to customize the workflow."""
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.allowed_type_cpp_map = {'<class \'_pyqcor.qreg\'>': 'qreg',
                                     '<class \'float\'>': 'double', 'typing.List[float]': 'std::vector<double>',
                                     '<class \'int\'>': 'int',
                                     '<class \'_pyxacc.quantum.PauliOperator\'>': 'qcor::PauliOperator'}
        self.__dict__.update(kwargs)

        # Create the qcor just in time engine
        self._qjit = QJIT()

        # Get the kernel function body as a string
        fbody_src = '\n'.join(inspect.getsource(self.function).split('\n')[2:])

        # Get the arg variable names and their types
        self.arg_names, _, _, _, _, _, self.type_annotations = inspect.getfullargspec(
            self.function)

        # Users must provide arg types, if not we throw an error
        if not self.type_annotations or len(self.arg_names) != len(self.type_annotations):
            print('Error, you must provide type annotations for qcor quantum kernels.')
            exit(1)

        # Construct the C++ kernel arg string
        cpp_arg_str = ''
        for arg, _type in self.type_annotations.items():
            if str(_type) not in self.allowed_type_cpp_map:
                print('Error, this quantum kernel arg type is not allowed: ', str(_type))
                exit(1)
            cpp_arg_str += ',' + \
                self.allowed_type_cpp_map[str(_type)] + ' ' + arg
        cpp_arg_str = cpp_arg_str[1:]

        globalVarDecl = []
        # Get all globals currently defined at this stack frame
        globalsInStack = inspect.stack()[1][0].f_globals
        globalVars = globalsInStack.copy()
        importedModules = {}
        for key in globalVars:
            descStr = str(globalVars[key])
            # Cache module import and its potential alias
            # e.g. import abc as abc_alias
            if descStr.startswith("<module "):
                moduleName = descStr.split()[1].replace("'", "")
                importedModules[key] = moduleName
            else:
                # Import global variables:
                # Only support float atm
                if (isinstance(globalVars[key], float)):
                    globalVarDecl.append(key + " = " + str(globalVars[key]))

        # Inject these global declarations into the function body.
        separator = "\n"
        globalDeclStr = separator.join(globalVarDecl)

        # Handle common modules like numpy or math
        # e.g. if seeing `import numpy as np`, we'll have <'np' -> 'numpy'> in the importedModules dict.
        # We'll replace any module alias by its original name,
        # i.e. 'np.pi' -> 'numpy.pi', etc.
        for moduleAlias in importedModules:
            if moduleAlias != importedModules[moduleAlias]:
                aliasModuleStr = moduleAlias + '.'
                originalModuleStr = importedModules[moduleAlias] + '.'
                fbody_src = fbody_src.replace(
                    aliasModuleStr, originalModuleStr)

        # Create the qcor quantum kernel function src for QJIT and the Clang syntax handler
        self.src = '__qpu__ void '+self.function.__name__ + \
            '('+cpp_arg_str+') {\nusing qcor::pyxasm;\n' + \
            globalDeclStr + '\n' + fbody_src + "}\n"

        # Handle nested kernels:
        dependency = []
        for kernelName in self.__compiled__kernels:
            # Check that this kernel *calls* a previously-compiled kernel:
            # pattern: "<white space> kernel(" OR "kernel.adjoint(" OR "kernel.ctrl("
            kernelCall = kernelName + '('
            kernelAdjCall = kernelName + '.adjoint('
            kernelCtrlCall = kernelName + '.ctrl('
            if re.search(r"\b" + re.escape(kernelCall) + '|' + re.escape(kernelAdjCall) + '|' + re.escape(kernelCtrlCall), self.src):
                dependency.append(kernelName)
                

        self.__kernels__graph.addKernelDependency(
            self.function.__name__, dependency)
        sorted_kernel_dep = self.__kernels__graph.getSortedDependency(
            self.function.__name__)

        # Run the QJIT compile step to store function pointers internally
        self._qjit.internal_python_jit_compile(self.src, sorted_kernel_dep)
        self._qjit.write_cache()
        self.__compiled__kernels.append(self.function.__name__)
        return

    # Static list of all kernels compiled
    __compiled__kernels = []
    __kernels__graph = KernelGraph()

    def get_internal_src(self):
        """Return the C++ / embedded python DSL function code that will be passed to QJIT
        and the clang syntax handler. This function is primarily to be used for developer purposes. """
        return self.src

    def kernel_name(self):
        """Return the quantum kernel function name."""
        return self.function.__name__

    def translate(self, q: qreg, x: List[float]):
        """
        This method is primarily used internally to map Optimizer parameters x : List[float] to 
        the argument structure expected by the quantum kernel. For example, for a kernel 
        expecting (qreg, float) arguments, this method should return a dictionary where argument variable 
        names serve as keys, and values are corresponding argument instances. Specifically, the float 
        argument variable should point to x[0], for example. 
        """

        if [str(x) for x in self.type_annotations.values()] == ['<class \'_pyqcor.qreg\'>', '<class \'float\'>']:
            ret_dict = {}
            for arg_name, _type in self.type_annotations.items():
                if str(_type) == '<class \'_pyqcor.qreg\'>':
                    ret_dict[arg_name] = q
                elif str(_type) == '<class \'float\'>':
                    ret_dict[arg_name] = x[0]
            if len(ret_dict) != len(self.type_annotations):
                print(
                    'Error, could not translate vector parameters x into arguments for quantum kernel.')
                exit(1)
            return ret_dict
        elif [str(x) for x in self.type_annotations.values()] == ['<class \'_pyqcor.qreg\'>', 'typing.List[float]']:
            ret_dict = {}
            for arg_name, _type in self.type_annotations.items():
                if str(_type) == '<class \'_pyqcor.qreg\'>':
                    ret_dict[arg_name] = q
                elif str(_type) == 'typing.List[float]':
                    ret_dict[arg_name] = x
            if len(ret_dict) != len(self.type_annotations):
                print(
                    'Error, could not translate vector parameters x into arguments for quantum kernel.')
                exit(1)
            return ret_dict
        else:
            print('currently cannot translate other arg structures')
            exit(1)

    def extract_composite(self, *args):
        """
        Convert the quantum kernel into an XACC CompositeInstruction
        """
        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(self.arg_names):
            args_dict[arg_name] = list(args)[i]
        return self._qjit.extract_composite(self.function.__name__, args_dict)

    def observe(self, observable, *args):
        """
        Return the expectation value of <observable> with 
        respect to the state given by this qjit kernel evaluated 
        at the given arguments. 
        """
        program = self.extract_composite(*args)
        return internal_observe(program, observable)

    def openqasm(self, *args):
        """
        Return an OpenQasm string representation of this 
        quantum kernel.
        """
        kernel = self.extract_composite(*args)
        staq = xacc.getCompiler('staq')
        return staq.translate(kernel)

    def print_kernel(self, *args):
        """
        Print the QJIT kernel as a QASM-like string
        """
        print(self.extract_composite(*args).toString())

    def n_instructions(self, *args):
        """
        Return the number of quantum instructions in this kernel. 
        """
        return self.extract_composite(*args).nInstructions()

    def ctrl(self, *args):
        print('This is an internal API call and will be translated to C++ via the QJIT.\nIt can only be called from within another quantum kernel.')
        exit(1)

    def __call__(self, *args):
        """
        Execute the decorated quantum kernel. This will directly 
        invoke the corresponding LLVM JITed function pointer. 
        """
        # Create a dictionary for the function arguments
        args_dict = {}
        for i, arg_name in enumerate(self.arg_names):
            args_dict[arg_name] = list(args)[i]

        # Invoke the JITed function
        self._qjit.invoke(self.function.__name__, args_dict)

        return


# Must have qpu in the init kwargs
# defaults to qpp, but will search for -qpu flag
init_kwargs = {'qpu': sys.argv[sys.argv.index(
    '-qpu')+1] if '-qpu' in sys.argv else 'qpp'}

# get shots if provided
if '-shots' in sys.argv:
    init_kwargs['shots'] = int(sys.argv[sys.argv.index('-shots')+1])

# Implements internal_startup initialization:
# i.e. set up qrt, backends, shots, etc.
Initialize(**init_kwargs)
