import yaml

# Load the YAML file
file_path = 'dvc.yaml'
with open(file_path, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

print(data)
