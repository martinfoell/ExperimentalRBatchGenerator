import statistics as st
import time
import sys
import torch
from torchmetrics import AUROC
from torchmetrics import ROC
from torchmetrics.classification import BinaryConfusionMatrix
import numpy as np
import ROOT
import _batchgenerator as RBG
def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)

def TestModel(model, gen_test):
    y_pred_all = torch.tensor([])
    y_target_all = torch.tensor([])    
    # Evaluate the model on the validation set
    for i, (x_train, y_train) in enumerate(gen_test):
        pred = model(x_train)
        y_pred = torch.flatten(pred)
        y_target = torch.flatten(y_train)        
        y_pred_all = torch.cat((y_pred_all, y_pred))
        y_target_all = torch.cat((y_target_all, y_target))
    
    return y_pred_all, y_target_all

def Metrics(pred_bkg, pred_sig, target_bkg, target_sig):
    pred = torch.cat((pred_bkg, pred_sig))
    target = torch.cat((target_bkg, target_sig))
    roc = ROC(task="binary")
    fpr, tpr, threshold = roc(pred, target.to(dtype=torch.int))
    auroc = AUROC(task="binary")
    auc = auroc(pred, target).item()
    return fpr, tpr, threshold, auc

num_epochs = 1
# chunks = 
chunks = float(sys.argv[1])
chunk_ranges = float(sys.argv[2])
chunk_size = 200000
range_size = 50000
# range_size = int(chunk_size / chunk_ranges)
batch_size = 1000
validation_split = 0.25
shuffling = True

epochs = 4

neuron = 70

AUC = []
D_SIG = []
S_val_ratio = []
n_trainings = int(sys.argv[3])

for k in range(n_trainings):
    rdf = ROOT.RDataFrame("tree", ["../data/Higgs_800_bkg.root", "../data/Higgs_800_sig.root"])
    # rdf.Snapshot("tree", "../data/both_sig_bkg.root")
    # rdf = ROOT.RDataFrame("tree", "../data/both_sig_bkg.root")
    # rdf = rdf.Define("Entry", "rdfentry_")

    n_signal = rdf.Filter("Label == 1").Count().GetValue()
    columns = rdf.GetColumnNames()
    target = ["Label"]    
    
    # gen_train, gen_validation =  ROOT.TMVA.Experimental.CreatePyTorchGenerators(rdf, batch_size, chunk_size, target=target, shuffle=shuffling, validation_split=0.25)
    gen_train, gen_validation =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffling)

    input_columns = gen_train.train_columns
    num_features = len(input_columns)

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
    
    size_train = 0
    size_val = 0    
    sig_train = 0
    sig_val = 0
    for i, (x_train, y_train) in enumerate(gen_train):
        size_train += y_train.size()[0]
        sig_train += torch.sum(y_train)
        print(torch.sum(y_train))
    
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
    
    n_bins = 30
    
    hist_bkg = ROOT.TH1F("hist_bkg", "Signal vs. Background, Affine PNN", n_bins, 0, 1)
    hist_sig = ROOT.TH1F("hist_sig", "Signal", n_bins, 0, 1)
    # confmat = ConfusionMatrix(task="binary", num_classes=2)
    bcm = BinaryConfusionMatrix()
    print(bcm(y_pred_all, y_true_all))
    PredSig = sum(y_pred_all[y_true_all == 1].detach().cpu().numpy())
    PredBkg = sum(y_pred_all[y_true_all == 0].detach().cpu().numpy())
    print(PredSig / 20000)
    print(PredBkg / 20000)
    # Fill histograms with the data
    for value in y_pred_all[y_true_all == 0].detach().cpu().numpy():
        hist_bkg.Fill(value)
        
    for value in y_pred_all[y_true_all == 1].detach().cpu().numpy():    
        hist_sig.Fill(value)
            
    hist_bkg.SetFillStyle(3001)  # Translucent style
    hist_bkg.SetFillColorAlpha(ROOT.kBlue,0.2)
    
    hist_sig.SetFillStyle(3001)  # Translucent style
    hist_sig.SetFillColorAlpha(ROOT.kRed,0.2)
                            
    # Create a canvas to draw the histograms
    canvas = ROOT.TCanvas("canvas", "Canvas", 1000, 600)
    # canvas.SetTitle("Background and Signal Comparison")
    # canvas.SetLogy()
    hist_bkg.SetStats(0)
    # Draw the histograms on the same canvas
    hist_bkg.Draw("HIST")  # Draw the background histogram
    hist_sig.Draw("HIST SAME")  # Draw the signal histogram on top of the background

    hist_bkg.GetXaxis().SetTitle("Neural Network Output")
    hist_bkg.GetXaxis().SetTitleOffset(1.3)  # Offset to adjust the position of the label (centered)
    hist_bkg.GetYaxis().SetTitle("Events")
    hist_bkg.GetYaxis().SetTitleOffset(1.3)  # Offset to adjust the position of the label (centered)

    # Add a legend
    legend = ROOT.TLegend(0.65, 0.6, 0.8, 0.85)  # (x1, y1, x2, y2)
    legend.AddEntry(hist_bkg, "Background", "f")
    legend.AddEntry(hist_sig, "m_{X} = 500")
    legend.Draw()

    print(r_sig_val)
    print(r_bkg_val)
    
    canvas.Update()
    canvas.Draw()

    canvas.SaveAs("SigBkg_AffPNN.png")
    input()
    sys.exit()
    canvas_roc = ROOT.TCanvas("c1", "TGraph Example", 1000, 800)

    roc_500 = ROOT.TGraph(len(fpr_500.cpu().numpy()), fpr_500.cpu().numpy(), tpr_500.cpu().numpy())

    roc_500.SetTitle("ROC curves, Affine PNN;X axis title;Y axis title;Z axis title");
    roc_500.SetLineWidth(2)

    # Update the canvas and display
    roc_500.Draw("AL")

    roc_500.GetXaxis().SetLimits(0, 1)  # Set x-axis range from 0 to 1
    roc_500.GetYaxis().SetRangeUser(0, 1)

    roc_500.SetLineColor(ROOT.kRed)

    roc_500.GetXaxis().SetTitle("Background efficency")
    roc_500.GetXaxis().SetTitleOffset(1.3)  # Offset to adjust the position of the label (centered)
    roc_500.GetYaxis().SetTitle("Signal efficency")
    roc_500.GetYaxis().SetTitleOffset(1.3)  # Offset to adjust the position of the label (centered)

    # Create a legend and add the graphs
    legend = ROOT.TLegend(0.6, 0.2, 0.85, 0.5)  # Define the position of the legend
    # legend.SetHeader("ROC Curves")  # Optional header for the legend

    # Add entries to the legend for each graph
    legend.AddEntry(roc_500, "m_{X} = 500, AUC = " + f"{round(auc_500,2)}", "l")
    # Draw the legend
    legend.Draw()


    canvas_roc.Update()
    canvas_roc.Draw()
    canvas_roc.SaveAs("roc_AffPNN.png")
    input()

# plt.figure(3)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_np, tpr_np)
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend(loc='lower right')
# plt.show()

#     print(D_sig)
#     print(r_sig_train)
#     print(r_bkg_train)
#     print(r_sig_val)
#     print(r_bkg_val)
#     print(chunk_size/range_size)
# S_val_ratio_round = []
# for i in S_val_ratio:
#     rounded = round(i, 4)
#     S_val_ratio_round.append(rounded)


# S_val_ratio_set = list(set(S_val_ratio_round))
# S_val_ratio_set.sort()

# Count = []
# for i in S_val_ratio_set:
#     count = S_val_ratio_round.count(i)
#     Count.append(count)

# PMF_est = []

# Tot_count = sum(Count)
# for i in range(len(Count)):
#     prob_i = Count[i]/Tot_count
#     PMF_est.append(prob_i)

# print(S_val_ratio_set)
# print(PMF_est)
    
# df = ROOT.RDF.FromNumpy({"S_val_ratio_set": np.array([S_val_ratio_set]),
#                              "PMF_est": np.array([PMF_est])})
# df.Snapshot("tree", f"../results/PMF/PMF_{n_trainings}_est_{chunks}_{chunk_ranges}.root")
    
    
