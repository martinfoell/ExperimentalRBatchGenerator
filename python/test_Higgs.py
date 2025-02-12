import time
import sys
import torch
from torchmetrics import AUROC
from torchmetrics import ROC
import numpy as np
import ROOT
import _batchgenerator as RBG

def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# rdf = ROOT.RDataFrame("tree", "../data/non-missing-variables.root")
rdf = ROOT.RDataFrame("tree", "../data/Higgs_800*.root")

n_signal = rdf.Filter("Label == 1").Count().GetValue()
print(rdf.Count().GetValue())
# sys.exit()
num_epochs = 1
chunk_size = 100000
range_size = 500
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

for test in range(1):
    size_train = 0
    size_val = 0    
    sig_train = 0
    sig_val = 0
    for i, (x_train, y_train) in enumerate(gen_train):
        size_train += y_train.size()[0]

        sig_train += torch.sum(y_train)
    
    for i, (x_val, y_val) in enumerate(gen_validation):
        size_val += y_val.size()[0]        
        sig_val += torch.sum(y_val)

r_sig_train = sig_train / size_train
r_bkg_train = (size_train - sig_train) / size_train
r_sig_val = sig_val / size_val
r_bkg_val = (size_val - sig_val) / size_val

D_sig = abs(r_sig_train - r_sig_val)

# sys.exit()

for epoch in range(epochs):

    ##############################################
    # Training
    ##############################################
    
    # Loop through the training set and train model
    model.train()
    tic = time.perf_counter()
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
        toc = time.perf_counter()

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
print(D_sig)
print(r_sig_train)
print(r_bkg_train)
print(r_sig_val)
print(r_bkg_val)
print(chunk_size/range_size)

data = [chunk_size, range_size, auc, D_sig]

# np_data = np.array(data)
# df = ROOT.RDF.FromNumpy({"chunk_size": np.array([chunk_size]),
#                          "range_size": np.array([range_size])})
# print(signal_train / size_train)
# print(signal_val / size_val)    
# print(signal_train + signal_val)

print(rdf.GetColumnNames())
