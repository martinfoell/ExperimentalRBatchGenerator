#include "../inc/RChunkLoader.hxx"
#include "../inc/RBatchLoader.hxx"




void batch() {

  ROOT::RDataFrame rdf("tree", "../data/file*.root");

  std::size_t chunkSize = 75;
  std::size_t rangeSize = 10;
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A"};
  bool shuffle = false;

  std::size_t batchSize = 10;
  std::size_t maxBatches = 50;  
  
  RChunkLoader<Double_t> chunkLoader(rdf, chunkSize,  rangeSize, validationSplit, columns, shuffle);
  
  TMVA::Experimental::RTensor<float> TrainTensor({0,0}); 
  TMVA::Experimental::RTensor<float> ValidationTensor({0,0}); 
  
  TMVA::Experimental::RTensor<float> TrainChunkTensor({0,0});
  TMVA::Experimental::RTensor<float> ValidationChunkTensor({0,0});   
  
  chunkLoader.PrintChunkDistributions();
  chunkLoader.PrintRangeDistributions();    
  
  chunkLoader.CreateRangeVector();
  chunkLoader.SortRangeVector();

  std::size_t numTrainChunks = chunkLoader.GetNumTrainChunks();
  std::size_t numValidationChunks = chunkLoader.GetNumValidationChunks();
  
  chunkLoader.CreateTrainRangeVector();
  chunkLoader.CreateValidationRangeVector();  
  chunkLoader.LoadTrainingDataset(TrainTensor);
  chunkLoader.LoadValidationDataset(ValidationTensor);    

  std::cout << "Train: " << TrainTensor.GetSize() << std::endl;
  std::cout << TrainTensor << std::endl;
  std::cout << " " << std::endl;        

  std::cout << "Validation: " << ValidationTensor.GetSize() << std::endl;
  std::cout << ValidationTensor << std::endl;
  std::cout << " " << std::endl;        
  
  for (int i = 0; i < numTrainChunks; i++) {
    chunkLoader.LoadTrainChunk(TrainChunkTensor, i);
    std::cout << "Train chunk " << i + 1 << ": " << TrainChunkTensor.GetSize() << std::endl;
    std::cout << TrainChunkTensor << std::endl;
  }


  std::cout << "Make batches " << std::endl;
  chunkLoader.LoadTrainChunk(TrainChunkTensor, 0);

  
  RBatchLoader batchLoader(chunkSize, batchSize, columns.size());

  // std::cout << *batchLoader.CreateBatch(TrainChunkTensor, 0) << std::endl;
  // std::cout << *batchLoader.CreateBatch(TrainChunkTensor, 1) << std::endl;
  // std::cout << *batchLoader.CreateBatch(TrainChunkTensor, 2) << std::endl;

  batchLoader.CreateTrainingBatches(TrainChunkTensor);

  chunkLoader.LoadTrainChunk(TrainChunkTensor, 1);  

  
  batchLoader.CreateTrainingBatches(TrainChunkTensor);
  batchLoader.CreateTrainingBatches(TrainChunkTensor);  
  
  std::cout << "Extract batche from queue " << std::endl;
  auto NumBatchQueue = batchLoader.GetNumTrainingBatchQueue();
  auto batch = batchLoader.GetTrainBatch();
  std::cout << batch << std::endl;
  batch = batchLoader.GetTrainBatch();
  std::cout << batch << std::endl;
  std::cout <<  NumBatchQueue << std::endl; 
  batch = batchLoader.GetTrainBatch();
  NumBatchQueue = batchLoader.GetNumTrainingBatchQueue();
  std::cout << batch << std::endl;
  std::cout <<  NumBatchQueue << std::endl;   
  batch = batchLoader.GetTrainBatch();
  std::cout << batch << std::endl;
  

  
  

  
}

