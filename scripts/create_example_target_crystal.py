from ase.build import bulk
from ase.io import write

target_crystal = bulk('Cu', 'fcc', a=3.6, cubic=True)

write("example_target.xsf", target_crystal)
