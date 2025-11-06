import sys
import os
from os.path import join, dirname
from dotenv import load_dotenv


script_dir = os.path.dirname(sys.modules['__main__'].__file__)
print(f'Script directory: {script_dir}')

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'Root directory: {root_dir}')
root_dir2 = os.path.abspath(os.curdir)
print(f'Root directory 2: {root_dir2}')


project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
print(f'Project directory: {project_dir}')


load_dotenv(os.path.join(root_dir, '.env'))
BASE_URL = os.environ.get("BASE_URL")
print(f'BASE_URL: {BASE_URL}')


path = '/dane/eutest/etl_process/data/consolidate_json/'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)














