# Clickbait classification

## Description
This project repository includes, all experiment carried out for clickbait classification for semeval2023 clickbait classification challenge. 

## Table of Contents
1. [Installation](#installation)
2. [Uasge](#usage)
4. [File discription](#file-discription)
3. [Contact Information](#contact-information)

## Installation

### Running on google colab
I am highly recommending to try and test python script on google colab, because of high RAM usage might cause local system crash. Moreover, all experiements were carried out using google colab environment. Therefore, there will be no additional requirement to install if script are run on google colab environment. 


#### steps to run script on google colab enviroment.
1. configuring user-name and email.


![configuring user-name and email address](https://github.com/Dip3102001/Clickbait-1/blob/main/SS/config_name_email.png)

2. cloning repository to google colab environment.


![cloning repo to colab](https://github.com/Dip3102001/Clickbait-1/blob/main/SS/cloning.png)

3. moving into clonned directory.


![chdir](https://github.com/Dip3102001/Clickbait-1/blob/main/SS/chdir.png)

4. running main.py script.


![running main.py](https://github.com/Dip3102001/Clickbait-1/blob/main/SS/main.png)
 
### Running on local system

#### Requirement 
In order to execute script on local system, it is require to meet following installation requirement. Installation of below dependencies differ from one OS to other. Please find steps on web to install below dependencies for Windows/Linux/MAC.

1. git
2. python
3. conda
4. pip

Once, above installation condition meet, we can start configuring environment for runtime.

1. cloning repo to local environment
```bash
# cloning repo
git clone 'https://github.com/Dip3102001/Clickbait-1'
```

2. configuring environment
```bash
# installing required package.
pip install -r 'requirement.txt'
```

3. running main.py 
```bash
# running main.py
python3 main.py
```
## Project structure

1. *Data Folder
- Data folder contains all training, evaluation and test data.

2. *outout Folder
- downloaded weights from the cloud and output generated via main.py is kept inside output folder.

3. *main.py
- main scirpt file to evaluate task-1 with different hyperparameter setting. This script is created for inference purpose only. 

4. *requirement.txt
- contains all Python packages that were used during project running.

5. *weights.pt
- This is human readable comma seperated file containing data about the location of weights file on cloud. extension '.pt' is bit misleading, rather it nothing but character human readable file.

 
## Usage

- main.py is python script file for evaluation of different task carried out during study for clickbait-classification. This python script file is interactive. Which asks for input from the user about which different hyperparameter setting one want to try. The reason why I made it interactive because there are in total more than 10 different hyperparameter setting configuration. Which makes it difficult to take input from command line and parse it down. 

## Contact Information

