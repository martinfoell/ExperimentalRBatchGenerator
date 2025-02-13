#include <TH2D.h>
#include <TCanvas.h>
#include <TAxis.h>
#include <vector>
#include <iostream>
#include <list>
#include <algorithm>

void plot_auc_diff_signal_frac() {
  gStyle->SetOptStat(0);
  gStyle->SetTitleFontSize(0.03);
  gStyle->SetLabelFont(43);
  gStyle->SetLabelSize(20);

  bool isAuc = false;
  ROOT::RDataFrame df("tree", "../results/merged/NN_output.root");
  
  auto n_ranges = *df.Take<Long64_t>("n_ranges");
  auto n_chunks = *df.Take<Long64_t>("n_chunks");
  auto auc_mean = *df.Take<Double_t>("auc_mean");
  auto auc_stdev = *df.Take<Double_t>("auc_stdev");
  auto D_sig_mean = *df.Take<Double_t>("D_sig_mean");
  auto D_sig_stdev = *df.Take<Double_t>("D_sig_stdev");    
  
  std::map<Long_t, int> chunk_map;

  chunk_map[3] = 1;
  chunk_map[6] = 2;
  chunk_map[12] = 3;
  chunk_map[24] = 4;

  std::map<Long_t, int> range_map;

  range_map[2] = 1;
  range_map[4] = 2;
  range_map[8] = 3;
  range_map[25] = 4;
  range_map[50] = 5;
  range_map[100] = 6;  
  
  std::vector<double> binsX = {1, 2, 3, 4, 5, 6, 7};
  std::vector<double> binsY = {1, 2, 3, 4, 5};

  TH2D *h = new TH2D("h", "", binsX.size() - 1, binsX.data(), binsY.size() - 1, binsY.data());

  double ranges = static_cast<double>(n_ranges[1]);
  double chunks = static_cast<double>(n_chunks[1]);
  
  
  TAxis *xAxis = h->GetXaxis();
  xAxis->SetBinLabel(1, "2");
  xAxis->SetBinLabel(2, "4");
  xAxis->SetBinLabel(3, "8");
  xAxis->SetBinLabel(4, "25");
  xAxis->SetBinLabel(5, "50");
  xAxis->SetBinLabel(6, "100");  


  TAxis *yAxis = h->GetYaxis();
  yAxis->SetBinLabel(1, "3");
  yAxis->SetBinLabel(2, "6");
  yAxis->SetBinLabel(3, "12");
  yAxis->SetBinLabel(4, "24");

  h->GetXaxis()->SetTitle("Ranges/Chunk");
  h->GetYaxis()->SetTitle("Chunks/Training set");
  
  h->GetXaxis()->SetLabelFont(43);
  h->GetXaxis()->SetLabelSize(20);
  h->GetYaxis()->SetLabelFont(43);
  h->GetYaxis()->SetLabelSize(20);
  h->SetTitleSize(0.3,"t");
  
  gStyle->SetPaintTextFormat(".2f");
  TCanvas *c = new TCanvas("c", "", 800, 600);

  for (int i = 0; i < n_ranges.size(); i++) {
    h->SetBinContent(range_map[n_ranges[i]], chunk_map[n_chunks[i]], auc_mean[i]);
    h->SetBinError(range_map[n_ranges[i]], chunk_map[n_chunks[i]], auc_stdev[i]);
  }
    
  h->SetTitle("AUC");
  double min_auc = *std::min_element(auc_mean.begin(), auc_mean.end());
  double max_auc = *std::max_element(auc_mean.begin(), auc_mean.end());
  h->GetZaxis()->SetRangeUser(min_auc, max_auc);
  
  h->SetMarkerSize(1.5);
  h->Draw("COLZ TEXTE"); 
  c->SaveAs("../plots/auc_grid.png");    
  TCanvas *c2 = new TCanvas("c2", "", 800, 600);
  
  for (int i = 0; i < n_ranges.size(); i++) {
    h->SetBinContent(range_map[n_ranges[i]], chunk_map[n_chunks[i]], D_sig_mean[i]*100);
    h->SetBinError(range_map[n_ranges[i]], chunk_map[n_chunks[i]], D_sig_stdev[i]*100);
  }
  h->SetTitle("#Delta s_{fraction} = #left| #frac{s_{train}}{s_{train}+b_{train}} - #frac{s_{val}}{s_{val}+b_{val}#right|}");
  double min_signal_frac = *std::min_element(D_sig_mean.begin(), D_sig_mean.end());
  double max_signal_frac = *std::max_element(D_sig_mean.begin(), D_sig_mean.end());
  h->GetZaxis()->SetRangeUser(min_signal_frac*100, max_signal_frac*100);

  h->SetMarkerSize(1.5);
  h->Draw("COLZ TEXTE");
  c2->SaveAs("../plots/diff_signal_fraction_grid.png");        
  
}
