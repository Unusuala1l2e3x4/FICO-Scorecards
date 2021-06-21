import json
# import pyahp # https://pyahp.gitbook.io/pyahp/
from pyahp import errors, hierarchy, methods, parse, parser, utils, validate_model

# print(dir(pyahp)) # [..., errors, hierarchy, methods, parse, parser, utils, validate_model]







# with open('model.json') as json_model:
#     # model can also be a python dictionary
#     model = json.load(json_model)

# ahp_model = parse(model)
# priorities = ahp_model.get_priorities()