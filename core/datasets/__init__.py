'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-09-26 00:32:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:32:41
'''
import importlib
import os

# automatically import any Python files in this directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        source = file[: file.find(".py")]
        module = importlib.import_module("core.datasets." + source)
        if "__all__" in module.__dict__:
            names = module.__dict__["__all__"]
        else:
            # import all names that do not begin with _
            names = [x for x in module.__dict__ if not x.startswith("_")]
        globals().update({k: getattr(module, k) for k in names})
