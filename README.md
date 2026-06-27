<div align="center">

# FUCrIMODo

<img src="./res/Fujimoto.jpg" width="200" height="200" alt="Fujimoto from the movie Ponyo">

**F**ind **U**nknown **C**rystals by **I**nversion of **M**L **O**ptimized **D**escriptors

---

</div>

**FUCrIMODo** is a program that aims to retrieve the crystal structure from
a SOAP descriptor.
This is done with the help of a novel *multi-stage* Genetic Algorithm (GA).

## Table of Contents

[TOC]

## Usage

### Install

The easiest way to install this program is through `conda`.
Enter the fucrimodo directory, where the `dependencies.yml` and the 
`pyproject.toml` is located and run:
```bash
conda env create -f ./environment.yml
```
This will install all dependencies and adds `fucrimodo` as a command line 
interface to the path in the conda env `fucrimodo-env`.
To now use the program run:
```bash
conda activate fucrimodo-env
```
In this environment fucrimodo can now be used as described below.

### Tutorials

#### Use fucrimodo as a library

If you want to use `fucrimodo` as a library, please refer to [this](tutorials/fucrimodo_as_library.ipynb) jupyter notebook. Even if you want to use fucrimodo exclusively as a cli tool it is recommended to do this tutorial to understand the api.

#### Use fucrimodo as standalone program

To use fucrimodo as a standalone program it is recommended to work with the `fucrimodo_lab`.
The `fucrimodo_lab` allows for simple management of configurations, data, analysis, ...
To set it up simply go to a desired directory (ideally outside the libraries git structure) and run:

``` bash
fucrimodo lab init
```

This sets up a directory called 'fucrimodo_lab' and sets up the required directory structure.
Exemplary raw data is provided to perform test runs.

....


### Documentation

Additional info can be found in the Documentation. To generate html docs
run the following:
```bash
cd docs/
make html
```

The docs will be generated at _build/html/ and can then be opened with the 
browser of your choice. E.g.:
```bash
firefox _build/html/index.html
```

## Customize

## Roadmap

MISSING

## Authors and acknowledgment

- Main Author: Me - Louis
- Co-Author: Martin Kuban

## License

MISSING

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
     (/ \\       |/     |/         /    \ `::'    
     `)  `\      o------o         '*~*~*~`         
                                   | //
                                        \ \\
                                         `.\\
                                           \((
```
MISSING CREDITS TO ARTISTS
