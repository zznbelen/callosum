# callosum
Belén corpus callosum register analysis code.

## Summary:
This code analyzes the recordings compound action potentials (CAP) of the corpus callosum and it:
a. Mesures the amplitude n1 and n2 and the area N1 and N2 of the CAP
b. Shows me the graph of the sweeps
c. Generates a spreadsheet (`.csv` file) with the valuse of several parameters including values of amplitudes n1, n2, and areas N1 and N2 of the CAP.

## Instructions:

- download code, e.g. to `~/Documents/GitHub/callosal`
- make docker container: inside the code directory, e.g. `~/Documents/GitHub/callosal`, run `make container`. This command creates a docker container with all the required libraries to run this code
- create a data directory, e.g. `~/Documents/GitHub/callosal/data`.
- Inside your data directory, create a subdirectory that contains your matlab files, e.g. `~/Documents/GitHub/callosal/data/20230723`. Note that the directory name as well as the matlab file names must not contain spaces or other special characters.
- create a results directory, e.g. `~/Documents/GitHub/callosal/results`. This is where the results (`.png` and `.csv` files) will be written to. All of the `.png` plots will be located in the subdirectory that has the same name as the corresponding data directory, for example `~/Documents/GitHub/callosal/results/20230723`, which will be created automatically when running the code.
- from the code directory, e.g. `~/Documents/GitHub/callosal`, run 'make' and specify the directory name of the data subdirectory. In our exmaple: `make 20230723`. This command will
  - launch the docker container
  - run the analysis code and write the results to the results directory
  - shut down the docker container

## Repository contents:
- `./`: contains the `Makefile` and some github files.
- `docker/`: contains files to build the docker container.
- `src/`: python source code as well as another `Makefile` to be run from within the docker container.

The main control file being called by the `Makefile` is `src/analyze_sweeps`, which, in turn, calls the library `src/sweeps.py`, that contains all of the analysis and plotting code.

## Prerequisites:
- docker
- command line terminal
