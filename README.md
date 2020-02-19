# ITU Research Project (Autumn 2019)

This repository contains the Research Project for the Autumn semester of 2019 by Rafał Markiewicz (rafm@itu.dk) and Sabin Rimniceanu (srim@itu.dk)

## Program Synthesis as Learning Rational Transductions

The subject area of this project focuses on applying Program Synthesis on string transduction tasks.  
String transduction is essentially a process of converting a string to another. It is widely used in Natural Language Processing to solve a range of tasks. This includes tasks such as spelling correction, grammatical error correction, grapheme to phoneme conversion, morphological inflection and even machine translation.

## Current state
At the moment, this project performs data preparation for proper analysis, 
as well as constructs decision trees that can be used to evaluate expected string operations 
given contexts (surrounding characters, grammar rules to apply)

### Prerequisites

* The project has been built and tested using Python 3.6. Please download the latest version [here](https://www.python.org/downloads/).
* If running from the command line or Python Console, the following `pip` packages are required:
    * biopython
    * more-itertools
    * numpy
    * pandas
    * regex
    * subword-nmt
    * sklearn
    * xlsxwriter
    * xlrd
    * openpyxl
    * matplotlib
    * graphviz
        * Additionally, the graphviz binaries must be installed to be able to create the decision tree charts. Follow instructions:
            * On Ubuntu (or Debian based distros):
                * run `sudo apt-get install graphviz`
            * On MacOS:
                * run `brew install graphviz`
            * On Windows
                * [download](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) the installer package and follow the instructions
                * Add `C:\Program Files (x86)\Graphviz2.38\bin` to User path
                * Add `C:\Program Files (x86)\Graphviz2.38\bin\dot.exe` to System Path
                * restart computer
    
### Running the project

The simplest way to run the program is by navigating to the project root and typing `python runner.py`

### Data

The `data` folder in the project root contains the following structure:  
|-`data`  
|--`all` - contains all language files  
|--`latin_alphabet` - contains only languages using the latin alphabet  
|--`processed` - pre-generated intermediate files used for the various data processing steps  

**NOTE**: Any of the above step files can be regenerated using the method in the `runner.py` file called:  
`write_steps(generate_step_1=False, generate_step_2=False, generate_step_3=False, generate_step_4=False, generate_step_5=False)`  
By setting any of the steps to true will generate the appropriate files in the corresponding folder in the `./data/processed` folder.
Further operations can be recreated using:
* `write_context_matrices()`
* `write_concepts()`
* `write_first_second_step_revision()`
