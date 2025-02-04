#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <iostream>




void create_tree()
{
   TFile f("bigfile.root", "RECREATE");
   TTree t("tree", "tree");
   t.SetAutoFlush(1000); // N is the number of entries per cluster
   double A;   
   double a;
   double b;
   double c;   
   t.Branch("A", &A);   
   t.Branch("a", &a);
   t.Branch("b", &b);
   t.Branch("c", &c);
   for (int i = 0; i < 50000000; ++i) {
      A = i + 100;      
      a = i + 200.1;
      b = i + 300.2;
      c = i + 400.3;
      t.Fill();
   }
   f.Write();
}
