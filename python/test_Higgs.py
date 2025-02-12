import sys
import torch
from torchmetrics import AUROC
from torchmetrics import ROC

import ROOT
import _batchgenerator as RBG

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# rdf = ROOT.RDataFrame("tree", "../data/non-missing-variables.root")
rdf = ROOT.RDataFrame("tree", "../data/Higgs_800*.root")

print(rdf.Count().GetValue())
# sys.exit()
num_epochs = 1
chunk_size = 200000
range_size = 100000
batch_size = 1000
columns = rdf.GetColumnNames()
target = ["Label"]
validation_split = 0.25
shuffle = True

gen_train, gen_validation =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

# Get a list of the columns used for training
input_columns = gen_train.train_columns
num_features = len(input_columns)
 
# sisy.exit()
 
neuron = 70

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
 
epochs = 10
 
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

    ##############################################
    # Validation
    ##############################################

    y_pred_all = torch.tensor([])
    y_true_all = torch.tensor([])    
    
    # Evaluate the model on the validation set
    model.eval()
    
    for i, (x_val, y_val) in enumerate(gen_validation):
        pred = model(x_val)
        
        y_pred = torch.flatten(pred)
        y_true = torch.flatten(y_val)        
        y_pred_all = torch.cat((y_pred_all, y_pred))
        y_true_all = torch.cat((y_true_all, y_true))        
        
        # Calculate accuracy
        accuracy = calc_accuracy(y_val, pred)
        print(f"Validation => accuracy: {accuracy}")        

roc = ROC(task="binary")
auroc = AUROC(task="binary")
fpr, tpr, thresholds = roc(y_pred_all, y_true_all.to(dtype=torch.int))
auc = auroc(y_pred_all, y_true_all).item()
print(auc)

print(rdf.GetColumnNames())
