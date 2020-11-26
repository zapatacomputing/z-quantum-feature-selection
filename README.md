# z-quantum-feature-selection
TODO TODO TODO
## What is it?

`z-quantum-feature-selection` is a library with functions for performing feature selection which can be used with  [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing computations on quantum computers.


## Usage

### Workflow
In order to use `z-quantum-feature-selection` in your workflow, you need to add it as an `import` in your Orquestra workflow:

```yaml
imports:
- name: z-quantum-feature-selection
  type: git
  parameters:
    repository: "git@github.com:zapatacomputing/z-quantum-feature-selection.git"
    branch: "master"
```

and then add it in the `imports` argument of your `step`:

```yaml
- name: my-step
  config:
    runtime:
      language: python3
      imports: [z-quantum-feature-selection]
```

Once that is done you can:
- use any `z-quantum-feature-selection` function by specifying its name and path as follows:
```yaml
- name: generate-qubo-for-feature-selection
  config:
    runtime:
      language: python3
      imports: [z-quantum-feature-selection]
      parameters:
        file: z-quantum-feature-selection/steps/qubo.py
        function: generate_qubo_for_feature_selection
```
- use tasks which import `zquantum.featureselection` in the python code (see below)

### Python

Here's an example of how to use methods from `z-quantum-feature-selection` in a python task:

```python
from zquantum.featureselection.qubo import generate_qubo_for_feature_selection
import numpy as np
x = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
y = np.array([1, 2])
alpha = 0.5
qubo = generate_qubo_for_feature_selection(x, y, alpha)
```

Even though it's intended to be used with Orquestra, `z-quantum-feature-selection` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory.

## Development and Contribution

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.
