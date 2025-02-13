import ROOT

df = ROOT.RDataFrame("tree", "../results/individual/*.root")

df = df.Define("n_chunks", "600000/chunk_size")
df = df.Define("n_ranges", "chunk_size/range_size")
df.Snapshot("tree", "../results/merged/NN_output.root")
