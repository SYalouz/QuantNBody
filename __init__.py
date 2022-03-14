# This file makes sure that when Quant_NBody is imported as a library correct functions are imported.
# If there is no "pybind/Quant_NBody_accelerate.so" file then it saves functions from Quant_NBody.
# On the other hand if the file is there fast functions are loaded. But we can still have access to slow
# ones through Quant_NBody.Quant_NBody.____.

import os.path
import os
compiled_library = os.path.abspath(__file__[:-12] + '/pybind/Quant_NBody_accelerate.so')
print(f'searching in {compiled_library}')
if os.path.isfile(compiled_library):
    print('Importing version of Quant_NBody with cpp (fast one)')
    from Quant_NBody.pybind.Quant_NBody_fast import *
    import Quant_NBody.pybind.Quant_NBody_fast
    import Quant_NBody.Quant_NBody
else:
    print(f"Couldn't find compiled c++ library on {compiled_library} so importing version of Quant_NBody with numba's"
          f" njit. This version is slower so you might consider installing pybind11 and accelerating calculations")
    from Quant_NBody.Quant_NBody import *
    import Quant_NBody.Quant_NBody
