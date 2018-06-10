#Readme
This is a Python program for handwrite number recognition. It is based on Tensorflow but the train data set is not MNIST set.
Instead of Tensorflow's MNIST, this program use the data provided on Kaggle's competition.
Link: https://www.kaggle.com/c/digit-recognizer
The CNN network gets 0.9902 accuracy on Kaggle platform at the end.
##How to execute it?
Make sure you have installed tensorflow and Python3 in your computer. Then just simply use this command in terminal:
### For numCNN.py
This program is based on CNN model as the file name.
- 'python numCNN.py 0' : It can use the model train before, which is saved in checkpoints, to make prediction on test.csv and then saved the result into output.csv.
- 'python numCNN.py 1' : Read in train.csv, and use the 28 * 28 pixel gray information in it to train the CNN model. And the result as a checkpoint which can be loaded later.
### For numLSTM.py
This program is based on LSTM model. Because it is just used as a baseline, so just use: 'python numLSTM.py'. And the train process will be showed and can be compared to CNN model.
## Folders
- checkpoint: checkpoints of CNN model is saved here.
- lstmChckpoint: checkpoints of LSTM model is saved here.
- SummaryCNN: the log for tensorboard is saved here. 
