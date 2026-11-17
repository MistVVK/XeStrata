"""Print every public and private top-level name of a module and the type of what it is bound to, one per line, sorted.
    python -I names_of.py <folder on sys.path> <module name>   (run it on the tree before and after a split and diff the outputs)
A moved function must still be reachable as module.name with the same kind of object."""
import importlib
import sys

sys.path.insert(0, sys.argv[1])
m = importlib.import_module(sys.argv[2])
for n in sorted(vars(m)):
    if n.startswith("__") and n.endswith("__"):
        continue
    v = getattr(m, n)
    print("%s %s" % (n, type(v).__name__))
