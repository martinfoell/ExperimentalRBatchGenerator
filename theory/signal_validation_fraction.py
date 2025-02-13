import ROOT
import numpy as np
# fileName = "../data/atlas-higgs-challenge-2014-v2.csv"

# df = ROOT.RDF.FromCSV(fileName)
# df = df.Filter("KaggleSet != 'u'")


# df = df.Redefine('Label', 'Label == "s" ? 1 : (Label == "b" ? 0 : -1)')


# df_bkg = df.Filter("Label == 0")
# df_sig = df.Filter("Label == 1")

# print(df_bkg.Count().GetValue())
# print(df_sig.Count().GetValue())

val_split = 0.25
n_events = 800000
s_events = 273375
b_events = 526625

n_chunks = 8
n_chunk_ranges = 4
n_val_chunks = val_split*n_chunks
n_train_chunks = (1 - val_split)*n_chunks

n_val_ranges = n_val_chunks * n_chunk_ranges
n_train_ranges = n_train_chunks * n_chunk_ranges
print(n_val_chunks, n_val_ranges)
print(n_train_chunks, n_train_ranges)
n_ranges = 32

n_s_ranges = np.floor(n_ranges*s_events/n_events)
n_b_ranges = np.ceil(n_ranges*b_events/n_events)

print(n_s_ranges)
print(n_b_ranges)


