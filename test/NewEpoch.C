#include "../inc/RChunkLoader.hxx"



void NewEpoch() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");

  std::size_t chunkSize = 50;
  std::size_t rangeSize = 25;
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A"};
  bool shuffle = false;
  
  RChunkLoader<Double_t> loader(rdf, chunkSize,  rangeSize, validationSplit, columns, shuffle);
  
  TMVA::Experimental::RTensor<float> TrainTensor({0,0}); 
  TMVA::Experimental::RTensor<float> ValidationTensor({0,0}); 
  
  TMVA::Experimental::RTensor<float> TrainChunkTensor({0,0});
  TMVA::Experimental::RTensor<float> ValidationChunkTensor({0,0});   
  
  loader.PrintChunkDistributions();
  loader.PrintRangeDistributions();    
  
  loader.CreateRangeVector();
  loader.SortRangeVector();

  std::size_t numTrainChunks = loader.GetNumTrainChunks();
  std::size_t numValidationChunks = loader.GetNumValidationChunks();
  
  loader.CreateTrainRangeVector();
  loader.CreateValidationRangeVector();  
  loader.LoadTrainingDataset(TrainTensor);
  loader.LoadValidationDataset(ValidationTensor);    

  std::cout << "Train: " << TrainTensor.GetSize() << std::endl;
  std::cout << TrainTensor << std::endl;
  std::cout << " " << std::endl;        

  std::cout << "Validation: " << ValidationTensor.GetSize() << std::endl;
  std::cout << ValidationTensor << std::endl;
  std::cout << " " << std::endl;        
  
  for (int i = 0; i < numTrainChunks; i++) {
    loader.LoadTrainChunk(TrainChunkTensor, i);
    std::cout << "Train chunk " << i + 1 << ": " << TrainChunkTensor.GetSize() << std::endl;
    std::cout << TrainChunkTensor << std::endl;
  }

  std::cout << " " << std::endl;        

  for (int i = 0; i < numValidationChunks; i++) {
    loader.LoadValidationChunk(ValidationChunkTensor, i);  
    std::cout << "Validation chunk " << i + 1 << ": " << ValidationChunkTensor.GetSize() << std::endl;
    std::cout << ValidationChunkTensor << std::endl;
    
  }

  std::cout << " " << std::endl;        
  std::cout << "============== New Epoch =============" << std::endl;
  std::cout << " " << std::endl;        
  
  // new epoch
  loader.CreateTrainRangeVector();
  loader.CreateValidationRangeVector();
  
  loader.LoadTrainingDataset(TrainTensor);
  loader.LoadValidationDataset(ValidationTensor);    

  std::cout << "Train: " << TrainTensor.GetSize() << std::endl;
  std::cout << TrainTensor << std::endl;
  std::cout << " " << std::endl;        

  std::cout << "Validation: " << ValidationTensor.GetSize() << std::endl;
  std::cout << ValidationTensor << std::endl;
  std::cout << " " << std::endl;        
  
  for (int i = 0; i < numTrainChunks; i++) {
    loader.LoadTrainChunk(TrainChunkTensor, i);
    std::cout << "Train chunk " << i + 1 << ": " << TrainChunkTensor.GetSize() << std::endl;
    std::cout << TrainChunkTensor << std::endl;
  }

  std::cout << " " << std::endl;        

  for (int i = 0; i < numValidationChunks; i++) {
    loader.LoadValidationChunk(ValidationChunkTensor, i);  
    std::cout << "Validation chunk " << i + 1 << ": " << ValidationChunkTensor.GetSize() << std::endl;
    std::cout << ValidationChunkTensor << std::endl;
    
  }
  

}
