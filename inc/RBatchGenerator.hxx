#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "RChunkLoader.hxx"
#include "RBatchLoader.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"
#include "TROOT.h"


template <typename... Args>
class RBatchGenerator {
 private:

  std::vector<std::string> fCols;
   
  std::size_t fChunkSize;
  std::size_t fMaxChunks;
  std::size_t fBatchSize;
  std::size_t fRangeSize;  
  std::size_t fNumColumns;
  std::size_t fNumEntries;

  float fValidationSplit;

  std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;
  std::unique_ptr<RBatchLoader> fBatchLoader;

  std::size_t fChunkNum;
  bool fShuffle;

  ROOT::RDataFrame &f_rdf;

  TMVA::Experimental::RTensor<float> fTrainTensor; 
  TMVA::Experimental::RTensor<float> fTrainChunkTensor;
  
  TMVA::Experimental::RTensor<float> fValidationTensor; 
  TMVA::Experimental::RTensor<float> fValidationChunkTensor;
  

 public:
  RBatchGenerator(ROOT::RDataFrame &rdf, const std::size_t chunkSize, const std::size_t rangeSize, const std::size_t batchSize,
                  const float validationSplit, const std::vector<std::string> &cols, bool shuffle)
    : f_rdf(rdf),
      fCols(cols),      
      fChunkSize(chunkSize),
      fRangeSize(rangeSize),
      fBatchSize(batchSize),        
      fValidationSplit(validationSplit),
      fShuffle(shuffle),
      fNumColumns(cols.size()),
      fTrainTensor({0, 0}),
      fTrainChunkTensor({0, 0}),
      fValidationTensor({0, 0}),
      fValidationChunkTensor({0, 0})
  
  {

    fChunkLoader = std::make_unique<RChunkLoader<Args...>>(f_rdf, fChunkSize, fRangeSize, fValidationSplit, fCols, fShuffle);
    fBatchLoader = std::make_unique<RBatchLoader>(fChunkSize, fBatchSize, fNumColumns);
    
    fChunkLoader->PrintChunkDistributions();
    fChunkLoader->PrintRangeDistributions();    

    fChunkLoader->CreateRangeVector();
    fChunkLoader->SortRangeVector();
    fChunkLoader->CreateTrainRangeVector();
    fChunkLoader->CreateValidationRangeVector();      
    
    fChunkLoader->LoadTrainingDataset(fTrainTensor);
    // fChunkLoader->LoadValidationDataset(fValidationTensor);        

    std::cout << "Train: " << fTrainTensor.GetSize() << std::endl;
    std::cout << fTrainTensor << std::endl;
    std::cout << " " << std::endl;        

    fChunkNum = 0;    
  }

  
  TMVA::Experimental::RTensor<float> GenerateTrainBatch() {
    auto batchQueue = fBatchLoader->GetNumTrainingBatchQueue();
    std::cout << "Batches in queue: " << batchQueue << std::endl; 
    if (batchQueue < 2) {
      std::cout << " " << std::endl;
      fChunkLoader->LoadTrainChunk(fTrainChunkTensor, fChunkNum);
      std::cout << "Loaded chunk: " << fChunkNum + 1 << std::endl;
      std::cout << fTrainChunkTensor << std::endl;
      std::cout << " " << std::endl;
      fBatchLoader->CreateTrainingBatches(fTrainChunkTensor);          
      fChunkNum++;
    }
    // Get next batch if available
    return fBatchLoader->GetTrainBatch();
  }

  /// \brief Returns the next batch of validation data if available.
  /// Returns empty RTensor otherwise.
};
