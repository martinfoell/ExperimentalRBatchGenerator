import ROOT
import _batchgenerator as RBG

rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf_node = ROOT.RDF.AsRNode(rdf)


num_epochs = 1
chunk_size = 110
range_size = 12
batch_size = 12
columns = ["A", "a"]
target = ["A"]
validation_split = 0
shuffle = False
# basegenerator = RBG.BaseGenerator(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

Print = False
gen_train =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

for j in range(num_epochs):
    print(f" ============== New Epoch {j+1} ==================")
    for i, (x_train, y_train) in enumerate(gen_train):
        if Print:
            print("Batches proces.: ", i)
# print(basegenerator.get_template(rdf_node))

# for i in range(5):
#     batch = basegenerator.GetTrainBatch()
#     torch_batch = basegenerator.ConvertBatchToPyTorch(batch)
#     print("Batch ", i)
#     print(torch_batch[0])
#     print(torch_batch[1])
