#include "../inc/RSplitTrainValidation.hxx"



void SplitTrainValidation() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");

  std::cout << rdf.Count().GetValue() << std::endl;

  float validationSplit = 0.3;
  std::size_t chunkSize = 80;
  std::size_t rangeSize = 20;  
  
  RSplitTrainValidation spltTrainValidation(rdf, chunkSize,  rangeSize, validationSplit);
  
}
