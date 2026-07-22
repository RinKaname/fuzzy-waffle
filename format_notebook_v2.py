import json

with open("solution.ipynb", "r") as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'def evaluate_threshold(' in "".join(cell['source']):
        source = cell['source']
        for i in range(len(source)):
            if 'predict_action(probe, policy, threshold)' in source[i]:
                source[i] = source[i].replace('predict_action(probe, policy, threshold)', 'predict_action(probe, policy)')
        cell['source'] = source

with open("solution.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
