import ROOT
import torch
import _batchgenerator as RBG
import sys
import numpy as np

# adding Folder_2 to the system path
sys.path.insert(0, '../python')

# rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf = ROOT.RDataFrame("tree", "../data/256.root")

# rdf_r = rdf.Filter("A > 300", 'cut 1')
# report = rdf_r.Report()
# print(report.Print())
# print(rdf.Count().GetValue())
# sys.exit()

# import _batchgenerator as bg


chunk_size = 60
range_size = 10
batch_size = 60
num_epochs = 1

columns = ["entry", "entry", "entry"]
target = ["entry"]
validation_split = 0
# validation_split = 0.333333333
shuffling = True;

gen_train =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffling)
# basegenerator = bg.BaseGenerator(rdf, chunk_size, range_size, batch_size, columns, target, validation_split, False)
# print(basegenerator.get_template(rdf_node, ["A"]))
# print(basegenerator.get_template(rdf_node))
size_train = 0
size_val = 0    
sig_train = 0
sig_val = 0


Batches = []
for i in range(1):
    for i, (x_train, y_train) in enumerate(gen_train):
        size_train += y_train.size()[0]
        sig_train += torch.sum(y_train)
        # print(x_train)
        # print(y_train)
        # print(x_train.numpy().flatten().astype(int).tolist())
        batch = y_train.numpy().flatten().astype(int).tolist()
        print(len(batch))
        Batches.append(batch)
        # print(
    
    # for i, (x_val, y_val) in enumerate(gen_validation):
    #     size_val += y_val.size()[0]        
    #     sig_val += torch.sum(y_val)


print(size_train, size_val)

# for i in range(5):
#     batch = basegenerator.GenerateTrainBatch()
#     torch_batch = basegenerator.ConvertBatchToPyTorch(batch)
#     print("Batch ", i)
#     print(torch_batch[0])
#     print(torch_batch[1])
# A = [1,2]
# B = [3,4]

# A += B
# # A.append(B)
Entries = np.array(Batches).flatten().astype(int).tolist()
print(sorted(Entries))
print(len(Entries))
