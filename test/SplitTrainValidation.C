#include "../inc/RSplitTrainValidation.hxx"



void SplitTrainValidation() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");


  float validationSplit = 0.5;
  std::size_t chunkSize = 35;
  std::size_t rangeSize = 25;  
  
  RSplitTrainValidation splitTrainValidation(rdf, chunkSize,  rangeSize, validationSplit);

  splitTrainValidation.PrintProperties();
  splitTrainValidation.CreateRangeVector();
  splitTrainValidation.PrintRangeVector();
}
