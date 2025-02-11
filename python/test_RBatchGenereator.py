import ROOT
import _batchgenerator as RBG

rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf_node = ROOT.RDF.AsRNode(rdf)


num_epochs = 3
chunk_size = 100
range_size = 5
batch_size = 10
columns = ["A", "a"]
target = ["A"]
validation_split = 0
shuffle = False
# basegenerator = RBG.BaseGenerator(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

gen_train =  RBG.CreatePyTorchGenerators(rdf, num_epochs, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)

for j in range(num_epochs):
    print(" ============== New Epoch ==================")
    for i, (x_train, y_train) in enumerate(gen_train):
        print(i)
# print(basegenerator.get_template(rdf_node))

# for i in range(5):
#     batch = basegenerator.GetTrainBatch()
#     torch_batch = basegenerator.ConvertBatchToPyTorch(batch)
#     print("Batch ", i)
#     print(torch_batch[0])
#     print(torch_batch[1])
