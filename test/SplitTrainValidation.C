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
  // float validationSplit = 0.5;
  // std::size_t chunkSize = 35;
  // std::size_t rangeSize = 20;  


  float validationSplit = 0.5;
  std::size_t chunkSize = 50;
  std::size_t rangeSize = 20; // now working with 10
  
  std::vector<std::string> columns = {"A"};

  TMVA::Experimental::RTensor<float> TrainTensor({0,0}); 
  TMVA::Experimental::RTensor<float> ValidationTensor({0,0}); 
  

  bool shuffle = true;
  RSplitTrainValidation<Double_t> splitTrainValidation(rdf, chunkSize,  rangeSize, validationSplit, columns, shuffle);
  
  splitTrainValidation.PrintProperties();
  // splitTrainValidation.PrintRangeVector();
  
  splitTrainValidation.Start();
  splitTrainValidation.LoadTrainingDataset(TrainTensor);
  splitTrainValidation.LoadValidationDataset(ValidationTensor);  

  std::cout << "Train" << std::endl;
  std::cout << TrainTensor << std::endl;
  std::cout << "Validation" << std::endl;
  std::cout << ValidationTensor << std::endl;  

}
