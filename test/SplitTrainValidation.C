#include "../inc/RSplitTrainValidation.hxx"



void SplitTrainValidation() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");


  float validationSplit = 0.3;
  std::size_t chunkSize = 85;
  std::size_t rangeSize = 23;  
  
  RSplitTrainValidation splitTrainValidation(rdf, chunkSize,  rangeSize, validationSplit);

  splitTrainValidation.PrintProperties();
}
