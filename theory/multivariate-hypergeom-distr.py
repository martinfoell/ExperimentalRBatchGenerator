import numpy as np
from prettytable import PrettyTable
from scipy.stats import multivariate_hypergeom
import ROOT

events = 800000
s_events = 273375
b_events = 526625
val_split = 0.25
s_frac = s_events / events
b_frac = b_events / events

val_events = val_split * events
chunks = 16
chunk_ranges = 2

val_chunks = val_split*chunks
train_chunks = (1 - val_split)*chunks

ranges = chunks * chunk_ranges 
val_ranges = val_chunks * chunk_ranges
train_ranges = train_chunks * chunk_ranges

chunk_size = events / chunks
range_size = events / ranges

full_s_ranges = np.floor(s_events / range_size)
full_b_ranges = np.floor(b_events / range_size)


mix_ranges = 0
if (full_s_ranges + full_b_ranges == ranges):
    mix_ranges =  0
else:
    mix_ranges = 1

if (mix_ranges == 1):
    mix_s_range_size = s_events - full_s_ranges * range_size
    mix_b_range_size = b_events - full_b_ranges * range_size
else:
    mix_s_range_size = 0
    mix_b_range_size = 0

if (mix_ranges == 1):
    mix_s_range_frac = mix_s_range_size / range_size
    mix_b_range_frac = mix_b_range_size / range_size
else:
    mix_s_range_frac = 0
    mix_b_range_frac = 0
    
tab_tv = PrettyTable()
tab_tv.add_column("Type", ["Total", "Training", "Validation"], align='l', valign='t')
tab_tv.add_column('Chunks', [chunks, round(train_chunks), round(val_chunks)], align='l', valign='t')
tab_tv.add_column('Ranges/Chunk', [chunk_ranges, chunk_ranges, chunk_ranges], align='l', valign='t')
tab_tv.add_column('Ranges', [ranges, round(train_ranges), round(val_ranges)], align='l', valign='t')
tab_tv.add_column('Chunk size', [chunk_size, "-", "-"], align='l', valign='t')
tab_tv.add_column('Range size', [range_size, "-", "-"], align='l', valign='t')


tab_sb = PrettyTable()
tab_sb.add_column("Type", ["Total", "Signal", "Background"], align='l', valign='t')
tab_sb.add_column("Events", [events, s_events, b_events], align='l', valign='t')
tab_sb.add_column("Fraction", [1, s_frac, b_frac], align='l', valign='t')
tab_sb.add_column("Full ranges", [ranges, round(full_s_ranges), round(full_b_ranges)], align='l', valign='t')
tab_sb.add_column("Mix ranges", [mix_ranges, mix_ranges, mix_ranges], align='l', valign='t')
tab_sb.add_column("Mix range size", [range_size, mix_s_range_size, mix_b_range_size], align='l', valign='t')
tab_sb.add_column("Mix range fraction", [range_size, mix_s_range_frac, mix_b_range_frac], align='l', valign='t')


print(tab_tv)
print(tab_sb)

s_ranges = 0
# mix_ranges = 0
b_ranges = round(val_ranges) - s_ranges - mix_ranges

SUM = 0
Prob = []
S_ratio = []
for s in range(round(val_ranges)+1):
    if (s == val_ranges):
        k = 1
    else:
        k = 2
    for m in range(k):
        s_ranges = s
        mix_range = m
        b_ranges = round(val_ranges) - s_ranges - mix_range
        # print(full_b_ranges, full_s_ranges, mix_ranges, val_ranges)
        A = multivariate_hypergeom.pmf(x=[b_ranges, s_ranges, mix_range], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=round(val_ranges))
        SUM += A
        Prob.append(float(round(A,2)))
        s_ratio = (s * range_size + m * mix_s_range_size) / (val_events)
        S_ratio.append(float(round(s_ratio*100, 1)))
        # Prob.append((round(A,2)))
        print(s, m, ":", A*100)
print("===========")
print(SUM)        
print("===========")

a = multivariate_hypergeom.pmf(x=[2, 0, 0], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=2)
b = multivariate_hypergeom.pmf(x=[1, 1, 0], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=2)
c = multivariate_hypergeom.pmf(x=[1, 0, 1], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=2)
d = multivariate_hypergeom.pmf(x=[0, 1, 1], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=2)
e = multivariate_hypergeom.pmf(x=[0, 2, 0], m=[round(full_b_ranges), round(full_s_ranges), round(mix_ranges)], n=2)

print(a, b, c, d, e)
tot = a + b + c + d + e
print(tot)

# [0.0, 0.117, 0.125, 0.242, 0.25, 0.367, 0.375, 0.492, 0.5, 0.617, 0.625]
p = [0.02, 0.01, 0.07, 0.07, 0.24, 0.08, 0.26, 0.05, 0.13, 0.04, 0.03]

tab_pmd = PrettyTable()
tab_pmd.add_row(["s_ratio"] + S_ratio)
tab_pmd.add_row(["prob"] + Prob)
print(tab_pmd)
# Prob_l = Prob
# print(Prob_l, len(Prob_l))
# print(S_ratio)
# print(p)
# print(sum(p), len(p))

y_values = Prob
bin_labels = S_ratio
# Number of bins
n_bins = len(y_values)

# Create a histogram with `n_bins` and appropriate range (you can adjust the range as needed)
hist = ROOT.TH1F("hist", "Histogram from y-values", n_bins, 0, n_bins)

# Fill the histogram using SetBinContent
for bin_idx, y in enumerate(y_values, start=1):  # start=1 because ROOT bins are 1-indexed
    hist.SetBinContent(bin_idx, y)
    hist.GetXaxis().SetBinLabel(bin_idx, str(bin_labels[bin_idx - 1])) 
hist.SetFillColor(ROOT.kRed)
# Create a canvas to draw the histogram
canvas = ROOT.TCanvas("canvas", "Canvas for Histogram", 1000, 600)

# Draw the histogram
hist.Draw()
# hist.GetXaxis().SetRangeUser(11,35)
# Show the canvas
canvas.Update()
canvas.Draw()
input()
