# Ptychography Example

Probe/Object update kernels taken from PtyPy.

## Installation

```
pip install .
```

## Run unit tests

```
pytest tests/
```

## Theory for multi-modal probe/object update

$$O_k^{(l)} = \frac{\sum_k \sum_j P_{x+x_j}^{(k)*}\psi_{j,x+x_j}^{(k,l)}}{\sum_k \sum_j \left| P_{x+x_j}^{(k)} \right|^2}$$

$$P_k^{(k)} = \frac{\sum_l \sum_j O_{x-x_j}^{(l)*}\psi_{j,x}^{(k,l)}}{\sum_l \sum_j \left| O_{x-x_j}^{(l)} \right|^2}$$

where $$k$$ are the probe modes, $$l$$ are the object modes and $j$ is the index over the views.
