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

  // TMVA::Experimental::RTensor<float> ChunkTensor((std::vector<std::size_t>{150, columns.size()}));
  TMVA::Experimental::RTensor<float> TrainTensor({0,0});
  // TrainTensor = TrainTensor.Resize({{150, columns.size()}});  

  // TMVA::Experimental::RTensor<float> TrainTensor({0,0});
  // TrainTensor = TrainTensor.Resize({{150, columns.size()}});  
  
  TMVA::Experimental::RTensor<float> ChunkTensor({0,0});  
  // TMVA::Experimental::RTensor<float> ChunkTensor(((std::vector<std::size_t>{0,0})));
  // std::cout << ChunkTensor.Shape()[0] << std::endl;
  // ChunkTensor = ChunkTensor.Resize({{150, columns.size()}});
  RSplitTrainValidation<Double_t> splitTrainValidation(rdf, ChunkTensor, chunkSize,  rangeSize, validationSplit, columns);

  splitTrainValidation.PrintProperties();
  // splitTrainValidation.CreateRangeVector();
  splitTrainValidation.PrintRangeVector();
  // splitTrainValidation.PrintTrainValidationVector();
  splitTrainValidation.LoadDataset(TrainTensor);

  // std::cout << ChunkTensor << std::endl;
  std::cout << TrainTensor << std::endl;  

}
