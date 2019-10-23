"""
Ignore this file, as it incorporates bad programming practices
and it has proven to be useless.
"""
import importlib


"""
def importa(path: str, alias: str = None): 
    if not alias:
      alias = os.path.split(path)[1].replace('.py', '')
    spec = importlib.util.spec_from_file_location(alias, path) 
    exec(f'{alias} = importlib.util.module_from_spec(spec)', globals()) 
    exec(f'spec.loader.exec_module({alias})', globals()) 
"""

def importa(path: str, alias: str = None): 
    if not alias:
      alias = os.path.split(path)[1].replace('.py', '')
    spec = importlib.util.spec_from_file_location(alias, path) 
    _locals = locals()
    exec(f'{alias} = importlib.util.module_from_spec(spec)', globals(), _locals) 
    exec(f'spec.loader.exec_module({alias})', globals(), _locals) 


## Importar módulo que mide tiempos :
spec = importlib.util.spec_from_file_location('timing', "drive/My Drive/DCI-Net/timing.py")
time = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time)

