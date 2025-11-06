import os, sys
from os.path import dirname, join, abspath
print(abspath(join(dirname(__file__), '..')))

sys.path.append('..')

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
print(sys.path)
sys.path.remove('/data/permanent/eutest/tests/function_tests')
print(abspath(join(dirname(__file__), '..')))

print(sys.prefix)