#1. Environments:
Ubuntu 18.04, Python3.6, tensorflow-gpu==1.10.1, keras==2.2.5, scikit-learn==0.23.2

#2. Data acquisition
To run this script, you should download the dataset from http://doi.org/10.5281/zenodo.4263528. And put the code folder and data folder in the same dictionary.

#3. Run script
You can change to the code dicionary and simply run the main.sh script. The training of cl-ecg-net and cl-pcg-net will successively be
excuted and finally the process of Genetic Algorithm.
cd ./code
. main.sh

Or you can respectively excute the subprogram.
e.g.
python3
cd ./code/
python ./cl-ecg-net/train.py
python ./cl-pcg-net/train.py
python ./genetic/train.py

