# Fucrimodo Lab

Welcome to the **fucrimodo lab**. Here you can run, organize, and reproduce
FUCrIMODo experiments. The tutorial below shows how to create your
first target file and invert it to an atomic structure. 

For more tutorials and a detailed documentation check out the Fucrimodo Docs at
<https://fucrimodo.readthedocs.io/en/latest/index.html>.

This guide assumes you have already read the README of the main fucrimodo
library and that you are familiar with Genetic Algorithms.

## Generate Target Files

First, we need a target file containing the descriptor features that should be
inverted. Use the `utils` submodul of fucrimodo to convert a structure file
(.xyz) into a feature vector and store it into a fucrimodo target file (.json). 

``` bash
fucrimodo utils \
    -a atoms_path=data/raw/simple-target.xyz \
    -a save_path=data/raw/simple-target.json
```

Confirm that the target file was created:

``` sh
ls data/raw/simple-target.json
```

## Perform a Run

With the target file ready, you can perform a run using the `run` submodule of
fucrimodo.

Use the -c flag to specify a custom run config. If no config is provided,
fucrimodo uses the default config, defined in configs/run/default.py. For all
fucrimodo commands except init, you can use -c to load a custom config from a
file. The documentation explains how to create these files.

The following command performs a run using the config at
configs/run/test_run_config.py.

``` sh
fucrimodo run \
    -v \
    -c configs/run/test_run_config.py \
    -s data/results/ \
    -n test_run \
    data/raw/simple-target.json
```

This will take a minute. Results are stored under the save directory 
(“-s”), in a sub-directory named after the run name (“-n”). In this case they go to 
data/results/test_run. Inspect the generated output:

``` sh
ls dat/results/test_run
tree data/results/test_run
```

All output files are either human-readable or easy to process with common CLI
tools such as jq. 

## Analyse Run Results

Fucrimodo includes dedicated analysis tools. You can analyse a full run, a
single stage, or multiple runs at once. Multi-run analysis is covered in the
tutorials of the Documentation.

To get an overview of the first run:

``` sh
fucrimodo analyse run data/results/test_run/
```

Then select a row from the statistics table to inspect (e.g. row 0):

``` sh
fucrimodo analyse run -r 0 data/results/test_run/
```

This shows global statistics. To analyse fitness functions or
genetic operators used in a specific stage, specify the stage directory. For example,
to plot the first fitness statistic run:

``` sh
fucrimodo analyse stage -r 0 data/results/test_run/stage_1/
```
