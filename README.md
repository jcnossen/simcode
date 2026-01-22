
This is the code accompanying the paper "Modulation-enhanced high density localization microscopy enabled by deep learning", 
implementing a pytorch based pipeline for converting STORM/DNA-PAINT microscopy datasets into localizations, making use of illumination patterns to enhance resolution.

## Reconstruction of a simulated live acquisition

[Download avi](/moving_smlm_sim.mp4)

## Google Colab Notebooks

An example notebook running within Google Colab can be found here:
https://colab.research.google.com/github/jcnossen/simcode/blob/master/notebooks/colab_simulation_example.ipynb

Training a full model is prohibitively slow on Colab and best done off the cloud (RTX 3090 running for ~20 hours), but just as an example
https://colab.research.google.com/github/jcnossen/simcode/blob/master/notebooks/colab_training_example.ipynb

## Local install

To install locally, these are the required dependencies. Miniconda/Anaconda on Windows 11 was used for this.

```
conda create -n simcode_env python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate simcode_env
cd <where you cloned the simcode repository>
pip install -e .
```


### FastPSF dependency 
To run the conventional fitting pipeline as a comparison, we use an earlier developed package called [FastPSF](https://www.gitlab.com/jcnossen/fastpsf). To build the binaries for this, 
you'll need to:

- download [cmake](https://cmake.org/), this will generate the project files for visual studio. 
- install visual studio community 2022: https://visualstudio.microsoft.com/vs/
- install CUDA Toolkit (tested with 12.4, as that is the latest pytorch compatible version, but 11.8 very likely also works fine). Install in this order, so CUDA will register its Visual Studio extensions.
- in your directory of choice, build the binaries
```
git clone https://gitlab.com/jcnossen/fastpsf.git
cd fastpsf
cmake .
cmake --build . --config Release
```
- and install locally in your python environment (make sure the environment you created above is active)
```
cd python
pip install -e .
```

### Run the simulation example

A simulation example (same as the google colab, but for a local install) can be found [here](example/simulation_example.py). This will generate a simulated microtubule reconstruction.

