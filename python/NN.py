import ROOT
import torch
import _batchgenerator as RBG
import sys
import numpy as np

# adding Folder_2 to the system path
sys.path.insert(0, '../python')

# rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf = ROOT.RDataFrame("tree", "../data/256.root")
# rdf = rdf.Filter("entry % 3 == 0", "test filter")
rdf = rdf.Filter("entry % 4 == 0", "test filter")
# Rdf_r = rdf.Filter("A > 300", 'cut 1')
# report = rdf_r.Report()
# print(report.Print())
print(rdf.Count().GetValue())
# sys.exit()

# import _batchgenerator as bg


chunk_size = 12
range_size = 4
batch_size = 4

columns = ["entry", "entry", "entry"]
target = ["entry"]
validation_split = 0.5
# validation_split = 0.333333333
shuffling = True;

gen_train, gen_validation =  RBG.CreatePyTorchGenerators(rdf, chunk_size, range_size, batch_size, columns, target, validation_split, shuffling)
size_train = 0
size_val = 0    
sig_train = 0
sig_val = 0


for i in range(2):
    TrainingBatches = []
    ValidationBatches = []
    for i, (x_train, y_train) in enumerate(gen_train):
        size_train += y_train.size()[0]
        sig_train += torch.sum(y_train)
        # print(x_train)
        # print(y_train)
        # print(x_train.numpy().flatten().astype(int).tolist())
        batch = y_train.numpy().flatten().astype(int).tolist()
        # print(len(batch))
        TrainingBatches.append(batch)
        # print(
    
    for i, (x_val, y_val) in enumerate(gen_validation):
        size_val += y_val.size()[0]        
        sig_val += torch.sum(y_val)
        batch = y_val.numpy().flatten().astype(int).tolist()
        # print(x_val)
        ValidationBatches.append(batch)
        
    TrainingEntries = np.array(TrainingBatches).flatten().astype(int).tolist()
    ValidationEntries = np.array(ValidationBatches).flatten().astype(int).tolist()
    DatasetEntries = TrainingEntries + ValidationEntries

    print(" ")
    print("Training size: ", len(TrainingEntries))
    print(sorted(TrainingEntries))
    print(" ")
    print("Validation size: ", len(ValidationEntries))
    print(sorted(ValidationEntries))
    print(" ")
    print("Dataset size: ", len(DatasetEntries))
    print(sorted(DatasetEntries))
    print(" ")

    print(TrainingBatches)
    print(ValidationBatches)



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

# ValSet = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 202, 203, 204, 205, 206, 207, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 250, 251, 252, 253, 254, 255]

# # ValSet = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 176, 177, 178, 179, 180, 181, 182, 183, 200, 201, 202, 203, 204, 205, 206, 207, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255]


# ValDiff = set(ValidationEntries) - set(ValSet)

# print(ValDiff)
