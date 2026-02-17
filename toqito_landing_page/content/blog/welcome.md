---
title: "Welcome to toqito"
date: 2025-02-17
draft: false
description: "An introduction to toqito - an open-source Python library for quantum information theory"
tags: ["introduction", "quantum computing", "python"]
---

Welcome to **toqito** (Theory of Quantum Information Toolkit) - a Python library designed to make quantum information theory research more accessible and efficient!

## What is toqito?

toqito is an open-source Python library that provides tools for studying various objects and properties in quantum information theory. Whether you're a researcher, student, or developer working with quantum computing, toqito offers a comprehensive set of utilities for working with:

- **Quantum States**: Bell states, Werner states, and many more
- **Quantum Channels**: Tools for analyzing quantum channels and their properties
- **Quantum Measurements**: Implementation of various measurement schemes
- **Entanglement**: Functions for studying entanglement properties
- **Nonlocality**: Tools for exploring nonlocal phenomena in quantum mechanics

<br>

## Why toqito?

Quantum information theory involves complex mathematical operations and concepts. toqito simplifies these operations by providing:

- Well-tested, documented implementations
- Intuitive Python API
- Integration with popular scientific Python libraries (NumPy, SciPy)
- Active development and community support

## Getting Started

Install toqito using pip:

```bash
pip install toqito
```

Here's a simple example of creating a Bell state:

```python
from toqito.states import bell

# Create the Bell state |Φ⁺⟩
phi_plus = bell(0)
print(phi_plus)
```

## Learn More

- **Documentation**: [toqito.readthedocs.io](https://toqito.readthedocs.io/)
- **GitHub**: [github.com/vprusso/toqito](https://github.com/vprusso/toqito)
- **Tutorials**: Check out our tutorial series for in-depth examples

## Get Involved

toqito is open source and welcomes contributions! Whether you want to:

- Report bugs
- Suggest new features
- Contribute code
- Improve documentation

Visit our [GitHub repository](https://github.com/vprusso/toqito) to get started.

We're excited to have you as part of the toqito community!
