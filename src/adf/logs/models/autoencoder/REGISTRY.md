<!--
SPDX-FileCopyrightText: 2026 Deutsche Telekom AG

SPDX-License-Identifier: Apache-2.0
-->

In the `:class:LogAutoEncoder` we are loading the backbones using the `model_loader` function. This function
looks to a registry called `MODEL_REGISTRY` inside the file `:python:registry.py` and if the model is registered
it returns it with the parameters that are given.

To register a model into the `MODEL_REGISTRY` you need to follow 2 steps:

1. Add the decorator `add_to_logs_registry` to the model class that you want to add into the registry.
2. Since the `MODEL_REGISTRY` is created during runtime the class definitions will need to be created (or imported) to be available for use.

For the 2nd point (2) in order to trigger the `add_to_logs_registry` function you'll need to import the model.
As you will notice in the `:python:__init__.py` we have the following `import` statements:

```python
from .lstm import LSTMAutoEncoder
from .vae import LSTMVariationalAutoEncoder
```

So whenever you import anything that is under the `adf.logs.autoencoder` it automatically triggers the `adf.logs.autoencoder.__init__.py`
which in turn imports the models and the class decorator adds them to the registry.

