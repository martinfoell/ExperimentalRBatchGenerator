import ROOT
import torch
import _batchgenerator as RBG
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '../python')

rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf_node = ROOT.RDF.AsRNode(rdf)


# import _batchgenerator as bg


chunk_size = 49
range_size = 23
batch_size = 10
num_epochs = 1
columns = ["A", "a"]
target = ["A"]
validation_split = 0.333333333
shuffling = True;

gen_train, gen_validation =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffling)
# basegenerator = bg.BaseGenerator(rdf, chunk_size, range_size, batch_size, columns, target, validation_split, False)
# print(basegenerator.get_template(rdf_node, ["A"]))
# print(basegenerator.get_template(rdf_node))
size_train = 0
size_val = 0    
sig_train = 0
sig_val = 0


for i, (x_train, y_train) in enumerate(gen_train):
    size_train += y_train.size()[0]
    sig_train += torch.sum(y_train)
    # print(torch.sum(y_train))
    
for i, (x_val, y_val) in enumerate(gen_validation):
    size_val += y_val.size()[0]        
    sig_val += torch.sum(y_val)


print(size_train, size_val)

# for i in range(5):
#     batch = basegenerator.GenerateTrainBatch()
#     torch_batch = basegenerator.ConvertBatchToPyTorch(batch)
#     print("Batch ", i)
#     print(torch_batch[0])
#     print(torch_batch[1])
