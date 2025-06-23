# Distance-Based Neural Networks (GPC)
A novel approach to neural computation through geometric transformations.

## Abstract
This repository contains the research on distance-based interpretation of neural networks. A 4.2KB model containing only 34 position vectors in 16-dimensional space successfully performs arithmetic calculations up to 16 trillion without using traditional arithmetic operators.

## Key Results
* **Model Size**: 4.2KB (4,268 bytes)
* **Computational Range**: -10¹³ to 10¹³
* **Accuracy**: >99.99% within training range
* **Key Innovation**: All computations are performed as movements in geometric space

## Model
* `gpc.pt` (4.2KB)

## Paper
Full paper: https://zenodo.org/records/15722051

## Core Principles
1. **Intentional Overfitting**: Each number has a fixed position
2. **Position Fixing**: Knowledge never forgets
3. **Distance-Based Learning**: No arithmetic operators used
4. **Unidirectional Information Flow**: New knowledge doesn't affect existing knowledge

## Citation
If you use this work, please cite:

Jinuk Cha. (2025). Distance-Based Interpretation of Neural Networks: A Study of GPC with a 4.2KB Model. Zenodo. https://doi.org/10.5281/zenodo.15722051

BibTeX:
```bibtex
@software{cha2025gpc,
  author       = {Jinuk Cha},
  title        = {Distance-Based Interpretation of Neural Networks: A Study of GPC with a 4.2KB Model},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15722051},
  url          = {https://doi.org/10.5281/zenodo.15722051}
}
```

## Contact
jinuk.cha.finance@gmail.com

## License
MIT License - see LICENSE file for details
