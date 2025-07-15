
## Multi-Fidelity Multi-Objective Bayesian Optimization: An Output Space Entropy Search Approach


---

## Requirements

This code is implemented in Python and requires the following dependencies:

* [`sobol_seq`](https://github.com/naught101/sobol_seq) – for generating Sobol sequences
* [`platypus`](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus) – for multi-objective evolutionary algorithms
* [`scikit-learn`](https://scikit-learn.org/stable/modules/gaussian_process.html) – specifically `sklearn.gaussian_process` for GP modeling
* [`pygmo`](https://esa.github.io/pygmo2/install.html) – for parallel optimization algorithms

You can install the required packages using:

```bash
pip install sobol_seq platypus-opt scikit-learn pygmo
```
---
## Running MF-OSEMO


```bash
python main.py <function_names> <d> <seed> <initial_number> <total_iterations> <sample_number> <approximation>
```

Here's an example command you could run from bash:

```bash
python main.py branin,Currin 2 0 5 100 10 TG
```

Explanation of arguments:

1. `function_names`: names of the benchmark functions separated by a comma
2. `d`: number of input dimensions 
3. `seed`: random seed 
4. `initial_number`: number of initial of evaluations
5. `total_iterations`: number of BO iterations
6. `sample_number`: number of samples to use for entropy estimation 
7.  `approximation`: approximation used for the acquisition function TG or NI

---

### Citation
If you use this code in your academic work please cite our papers:
```bibtex

@inproceedings{belakaria2020multifidelity,
  title={Multi-fidelity multi-objective bayesian optimization: an output space entropy search approach},
  author={Belakaria, Syrine and Deshwal, Aryan and Doppa, Janardhan Rao},
  booktitle={Proceedings of the AAAI Conference on artificial intelligence},
  volume={34},
  number={06},
  pages={10035--10043},
  year={2020}
}
@article{belakaria2021output,
  title={Output Space Entropy Search Framework for Multi-Objective Bayesian Optimization},
  author={Belakaria, Syrine and Deshwal, Aryan and Doppa, Janardhan Rao},
  journal={Journal of Artificial Intelligence Research},
  volume={72},
  pages={667-715},
  year={2021}
}

````

