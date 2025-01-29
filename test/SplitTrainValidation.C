#include "../inc/RSplitTrainValidation.hxx"



void SplitTrainValidation() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");


  // Case i)
  // float validationSplit = 0.5;
  // std::size_t chunkSize = 35;
  // std::size_t rangeSize = 25;  


  // case ii)
  // float validationSplit = 0.3;
  // std::size_t chunkSize = 85;
  // std::size_t rangeSize = 15;  

  // case iii)
  // float validationSplit = 0.7;
  // std::size_t chunkSize = 85;
  // std::size_t rangeSize = 15;  
  
  // Case iv)
  float validationSplit = 0.5;
  std::size_t chunkSize = 35;
  std::size_t rangeSize = 20;  
  
  RSplitTrainValidation splitTrainValidation(rdf, chunkSize,  rangeSize, validationSplit);

  splitTrainValidation.PrintProperties();
  splitTrainValidation.CreateRangeVector();
  splitTrainValidation.PrintRangeVector();
}
