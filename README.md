# crr_prediction
Python software library implementing meta models of Bayesian Feed Forward Neural Networks (Bayesian-FFNN) and Bayesian Convolutional Neural Networks (Bayesian-CNN) that use Bayesian optimization techniques for automatic neural model selection.
The repository includes also  experiments for the detection of active cis-regulatory regions, i.e. promoters and enhancers, in specific cell lines.


## Notes on installing Ray
We need the very latest version of ray, that can be currently installed from
the tutorial available [here](https://docs.ray.io/en/master/installation.html).

You can identify your python version by running `python -V`.

If you get errors such as `the wheel is not supported` just update your pip installation by using:

```bash
pip install pip -U
```
