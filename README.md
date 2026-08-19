<div align="center">

# FUCrIMODo

<img src="./res/Fujimoto_legal.png" width="200" height="200" alt="Fujimoto from the movie Ponyo">

**F**ind **U**nknown **C**rystals by **I**nversion of **M**L **O**ptimized **D**escriptors

---

</div>

**FUCrIMODo** is a scientific framework for recovering atomic structures from
machine-learning descriptors. It is built around a novel multi-stage Genetic
Algorithm (GA). The method and program are introduced in [TODO: MISSING].
FUCrIMODo comes with an inversion algorithm for the global SOAP descriptor out
of the box, and more descriptors are on the way or can be added by you!

## Table of Contents

- [Requirements](#requirements)
- [Install](#install)
- [Tutorials](#tutorials)
  - [Use fucrimodo CLI](#use-fucrimodo-cli)
  - [Use fucrimodo as a library](#use-fucrimodo-as-a-library)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Authors and acknowledgment](#authors-and-acknowledgment)
- [License](#license)

## Requirements

- Python 3.12 or later
- [Numpy](https://numpy.org/doc/stable/) (Handle arrays and calculations.)
- [Pandas](https://pandas.pydata.org/docs/index.html) (Handle data.)
- [DEAP](https://deap.readthedocs.io/en/master/) (GA framework.)
- [ASE-GA](https://dtu-energy.github.io/ase-ga/) (Atomic structure GA Framework.)
- [Atomic Simulation Environment (ASE)](https://docs.ase-lib.org/index.html)(Atomic structure framework.)
- [PyXtal](https://pyxtal.readthedocs.io/en/latest/index.html) (Atomic structure sampling.)
- [MatID](https://singroup.github.io/matid/index.html#) (Perform Atomic symmetry operations.)
- [DScribe](https://singroup.github.io/dscribe/2.1.x/#) (Descriptor calculator.)
- [Click](https://click.palletsprojects.com/en/stable/) (CLI backend.)
- [Matplotlib](https://matplotlib.org/) (2D Plotting.)

## Install

To install the latest release:

``` bash
pip install fucrimodo
```

Or to install the development version:

``` bash
pip install git+git@github.com:OHANAN1/fucrimodo.git
```

For more detailed instructions, including setup with `uv` and `conda`, please refer to the documentation at [TODO: MISSING].

## Tutorials

### Use fucrimodo CLI

To use fucrimodo as a cli you need to set up a `fucrimodo_lab`.  The
`fucrimodo_lab` is a human-readable database that allows you to manage 
configurations, data, analysis and more. To set it up, go to a desired
directory (ideally outside the library's git structure) and run:

``` bash
fucrimodo lab init
```

This creates a directory called fucrimodo_lab and sets up the required directory
structure. Example raw data is provided so you can perform test runs. Please set
it up and refer to the README.md file inside the lab for more info.

### Use fucrimodo as a library

To learn how to configure the CLI tool or use fucrimodo as a library, you can work through [this Jupyter notebook tutorial](tutorials/fucrimodo_as_library.ipynb). Also refer to the documentation for more details.

## Documentation

The documentation is hosted at [TODO: MISSING]. It includes additional tutorials
and documents the API of fucrimodo.

To build it yourself, first install the dependencies:

``` bash
pip install ".[docs]"
```

Now an HTML version of the docs can be generated:

``` bash
cd docs/
make html
```

The docs will be generated at _build/html/ and can then be opened with the
browser of your choice. E.g.:

``` bash
qutebrowser _build/html/index.html
```


## Roadmap

- [ ] Implement and test additional descriptors types
- [ ] Implement new Stage types
    - [ ] `ParallelGAStage` (Run multiple GA stages parallel.)
    - [ ] `SwarmSearchStage` (Use a swarm search for the ideal descriptor.)
    - [ ] `GradientDescentStage` (Follow the descriptor gradients.)
- [ ] Improve current default run configuration for bigger structures
- [ ] Add a proper way to update `fucrimodo_lab` defaults without overwriting
      existing defaults.

## Contact

- GitHub issues: https://github.com/OHANAN1/fucrimodo/issues
- Email: louis.boehm@gmx.de

## Authors and acknowledgment

- Main Author: Louis Böhm
- Co-Author: Martin Kuban

## License

The program is licensed with the Apache 2.0 license.

## Citation

If you use this program in a scientific publication please add the following citation:
[TODO: MISSING]

## Little Reward

As a reward that you read the complete README.md file you can now look at this
cute ASCII-Art. :D
```txt
  (\{\             .               ,@@@@                
  { { \ ,~,  ^  .     ~        __ _ ),\\(\   _,::;
   {   \|`) <*>   +  o------o  .)\)\\_(((\),:::::;
  { {  /(\  /~      /|     /|   `\`._,)))))::::::`,
   {/{/; ,\/       o------o |     `.__/(((:::::::' 
      [[ '         | |    | |        \  (`:::::::.
       \` \        | o----+-o         @**\ `:::::; 
       (/ \\       |/mlp  |/         /    \ `::'    
ejm    `)  `\      o------o         '*~*~*~`         
                                      | //
                                      \ \\
                                       `.\\
                                         \((
                                          ` ` hjw
```
(`I will be a human, too!`~Ponyo)



