import yaml

x = {'a':1, 'b':2}

with open('./f.yaml', 'w') as file:
    yaml.dump(x, file, default_flow_style=False)