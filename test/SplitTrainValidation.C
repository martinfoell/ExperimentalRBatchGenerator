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



  std::size_t chunkSize = 100;
  std::size_t rangeSize = 50;
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A"};
  bool shuffle = false;
  
  RChunkLoader<Double_t> loader(rdf, chunkSize,  rangeSize, validationSplit, columns, shuffle);
  
  TMVA::Experimental::RTensor<float> TrainTensor({0,0}); 
  TMVA::Experimental::RTensor<float> ValidationTensor({0,0}); 
  
  TMVA::Experimental::RTensor<float> TrainChunkTensor({0,0});
  TMVA::Experimental::RTensor<float> ValidationChunkTensor({0,0});   
  
  loader.Start();

  loader.PrintChunkDistributions();
  loader.PrintRangeDistributions();    
  loader.LoadTrainingDataset(TrainTensor);
  loader.LoadValidationDataset(ValidationTensor);  


  std::cout << "Train: " << TrainTensor.GetSize() << std::endl;
  std::cout << TrainTensor << std::endl;
  std::cout << " " << std::endl;        

  std::cout << "Validation: " << ValidationTensor.GetSize() << std::endl;
  std::cout << ValidationTensor << std::endl;
  std::cout << " " << std::endl;      

  std::cout << "Checks:" << std::endl;
  std::cout << "Training: ";
  loader.CheckIfUnique(TrainTensor);
  std::cout << "Validation: ";
  loader.CheckIfUnique(ValidationTensor);
  std::cout << "Overlap between training and validation tensors: ";
  loader.CheckIfOverlap(TrainTensor, ValidationTensor);

  std::cout << " " << std::endl;        

  std::size_t numTrainChunks = loader.GetNumTrainChunks();

  // loader.SplitTrainRanges();
  // loader.CreateTrainRangeVector();
  // for (int i = 0; i < numTrainChunks; i++) {
  //   loader.LoadTrainChunk(TrainChunkTensor, i);
  //   std::cout << "Train chunk " << i + 1 << ": " << TrainChunkTensor.GetSize() << std::endl;
  //   std::cout << TrainChunkTensor << std::endl;
  //   // Shuffle training ranges function

  //   loader.LoadTrainChunk(TrainChunkTensor, i);      
  //   std::cout << "Train chunk " << i + 1 << ": " << TrainChunkTensor.GetSize() << std::endl;
  //   std::cout << TrainChunkTensor << std::endl;
    
  // }

  // std::cout << " " << std::endl;        

  // std::size_t numValidationChunks = loader.GetNumValidationChunks();

  // for (int i = 0; i < numValidationChunks; i++) {
  //   loader.LoadValidationChunk(ValidationChunkTensor, i);  
  //   std::cout << "Validation chunk " << i + 1 << ": " << ValidationChunkTensor.GetSize() << std::endl;
  //   std::cout << ValidationChunkTensor << std::endl;
    
  // }

}
