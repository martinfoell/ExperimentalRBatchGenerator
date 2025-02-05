import ROOT
import _batchgenerator as bg

rdf = ROOT.RDataFrame("tree", "../data/file*.root")
rdf_node = ROOT.RDF.AsRNode(rdf)


chunk_size = 100
range_size = 5
batch_size = 10
columns = ["A", "a"]
target = ["A"]
validation_split = 0.3
shuffle = False
basegenerator = bg.BaseGenerator(rdf, chunk_size, range_size, batch_size, columns, target, validation_split, shuffle)


print(basegenerator.get_template(rdf_node))

for i in range(5):
    batch = basegenerator.GenerateTrainBatch()
    torch_batch = basegenerator.ConvertBatchToPyTorch(batch)
    print("Batch ", i)
    print(torch_batch[0])
    print(torch_batch[1])
