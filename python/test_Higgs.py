import sys
from torchmetrics import AUROC
import torch

import ROOT
import _batchgenerator as RBG



# rdf = ROOT.RDataFrame("tree", "../data/non-missing-variables.root")
rdf = ROOT.RDataFrame("tree", "../data/Higgs*.root")

print(rdf.Count().GetValue())
# sys.exit()
num_epochs = 1
chunk_size = 500000
range_size = 500000
batch_size = 1000
columns = rdf.GetColumnNames()
target = ["Label_int"]
validation_split = 0
shuffle = True
# basegenerator = RBG.BaseGenerator(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

Print = True
gen_train =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

# Get a list of the columns used for training
input_columns = gen_train.train_columns
num_features = len(input_columns)
 
 
def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)
 
neuron = 30

# Initialize PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(num_features, neuron),
    torch.nn.BatchNorm1d(neuron),
    torch.nn.ReLU(),
    torch.nn.Linear(neuron, neuron),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(neuron),
    torch.nn.Linear(neuron, neuron),
    torch.nn.ReLU(),
    torch.nn.Linear(neuron, 1),
    torch.nn.Sigmoid(),
)

loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
 
epochs = 5
 
for epoch in range(epochs):

    ##############################################
    # Training
    ##############################################
    
    # Loop through the training set and train model
    model.train()
    for i, (x_train, y_train) in enumerate(gen_train):
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        
        # improve model
        model.zero_grad()
        loss.backward()
        optimizer.step()
 
        # Calculate accuracy
        accuracy = calc_accuracy(y_train, pred)
 
        print(f"Training => accuracy: {accuracy}")
            

print(rdf.GetColumnNames())
