import os
import importlib

from .vanilla_trainer import vanilla_train
from .odin_tester import odin_test
from .credal_trainer import credal_train


# -- Do it automatically --

# # trainers/__init__.py
# import os
# import importlib

# # Automatically import all modules
# package_dir = os.path.dirname(__file__)
# for filename in os.listdir(package_dir):
#     if filename.endswith(".py") and filename != "__init__.py":
#         module_name = filename[:-3]
#         module = importlib.import_module(f"{__name__}.{module_name}")
#         # Pull out all functions that end in `_train` and expose them
#         for attr_name in dir(module):
#             if attr_name.endswith("_train"):
#                 globals()[attr_name] = getattr(module, attr_name)
