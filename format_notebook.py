import json

with open("solution.ipynb", "r") as f:
    notebook = json.load(f)

# Find the first code cell that has the policy parser and predictor logic
target_cell = None
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'def parse_policy(' in "".join(cell['source']):
        target_cell = cell
        break

if target_cell:
    with open("phase_1_estimator.py", "r") as f:
        phase1_code = f.read()

    # We will replace the target cell's source with the updated code
    target_cell['source'] = phase1_code.split('\n')
    # add newlines back to the source lines as jupyter expects them
    target_cell['source'] = [line + '\n' for line in target_cell['source']]

    with open("solution.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)

    print("Notebook successfully updated with phase_1_estimator.py logic.")
else:
    print("Could not find the target cell to replace in the notebook.")
