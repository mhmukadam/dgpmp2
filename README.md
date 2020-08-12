dGPMP2
=====

This library is a PyTorch implementation of dGPMP2 algorithm published in [Differentiable Gaussian Process Motion Planning](https://bit.ly/33SGjms), ICRA 2020.


Installation
-----

- Install [Anaconda](https://www.anaconda.com/products/individual) (Python 3.7)
- Clone the repository and run the following steps
  ```bash
  conda create -n diff_gpmp2 python=3.7
  conda env update -n diff_gpmp2 -f environment.yml
  conda activate diff_gpmp2
  ```
- Install [OMPL](https://github.com/ompl/ompl) with [Python bindings](https://ompl.kavrakilab.org/python.html) (for generating data and expert trajectories) 


Usage
-----
- Dataset generation example
  ```bash
  cd diff_gpmp2/datasets
  sh generate_2d_dataset.sh
  ```
- Fully differentiable planning example 
  ```bash
  cd examples/
  python diff_gpmp2_2d_example.py
  ```


Questions & Bug reporting
-----

Please use Github issue tracker to report bugs.


Citing
-----

If you use this library in an academic context, please cite the following publication:

```
@article{bhardwaj2019differentiable,
  title={Differentiable {G}aussian process motion planning},
  author={Bhardwaj, Mohak and Boots, Byron and Mukadam, Mustafa},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```


License
-----

dGPMP2 is released under the BSD license. See LICENSE file.
