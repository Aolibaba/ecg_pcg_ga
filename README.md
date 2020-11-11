#1. Environments:
Ubuntu 18.04, Python3.6, tensorflow-gpu==1.10.1, keras==2.2.5, scikit-learn==0.23.2

#2. Data acquisition
To run this script, you should download the dataset from http://doi.org/10.5281/zenodo.4263528. Note that put the code folder and data folder in the same dictionary.

#3. Run script
You can change to the code dictionary and simply run the main.sh script. The training of cl-ecg-net and cl-pcg-net will successively be excuted and finally the process of Genetic Algorithm.
cd ./code
. main.sh

Or you can respectively excute the subprogram.
<<<<<<< HEAD
e.g. solely train the cl-ecg-net.
cd ./code/cl-ecg-net/
python train.py
=======
python3
cd ./code/
python ./cl-ecg-net/train.py
python ./cl-pcg-net/train.py
python ./genetic/train.py
>>>>>>> 08b4e6b8f356787ac367c7880a9fc690f1997e2f
