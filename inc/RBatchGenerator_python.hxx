#include <unistd.h>
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

  std::unique_ptr<std::thread> fLoadingThread;
  
  std::size_t fChunkNum;
  bool fShuffle;

  ROOT::RDataFrame &f_rdf;

  std::mutex fIsActiveMutex;
  bool fIsActive{false}; // Whether the loading thread is active
  bool fNotFiltered;
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
      fValidationChunkTensor({0, 0}),
      fNotFiltered(f_rdf.GetFilterNames().empty())      
  
  {

    fChunkLoader = std::make_unique<RChunkLoader<Args...>>(f_rdf, fChunkSize, fRangeSize, fValidationSplit, fCols, fShuffle);
    fBatchLoader = std::make_unique<RBatchLoader>(fChunkSize, fBatchSize, fNumColumns);
    
    fChunkLoader->PrintChunkDistributions();
    fChunkLoader->PrintRangeDistributions();    

    fChunkLoader->CreateRangeVector();
    fChunkLoader->SortRangeVector();
    fChunkLoader->CreateTrainRangeVector();
    fChunkLoader->CreateValidationRangeVector();      
    
    // fChunkLoader->LoadTrainingDataset(fTrainTensor);
    // // fChunkLoader->LoadValidationDataset(fValidationTensor);        

    // std::cout << "Train: " << fTrainTensor.GetSize() << std::endl;
    // std::cout << fTrainTensor << std::endl;
    // std::cout << " " << std::endl;        

    fChunkNum = 0;    
  }

  ~RBatchGenerator() { DeActivate(); }


   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = false;
      }

      fBatchLoader->DeActivate();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      if (fIsActive)
         return;

      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = true;
      }

      fBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
      if (fNotFiltered) {
        std::cout << "Not filtered" << std::endl;
        // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksNoFilters, this);
      } else {
        std::cout << "Filtered: needs to be implemented" << std::endl;        
        // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksFilters, this);
      }
   }
  
   /// @brief Load chunks when no filters are applied on rdataframe
   // void LoadChunksNoFilters()
   // {
   //    for (std::size_t currentChunk = 0, currentEntry = 0;
   //         ((currentChunk < fMaxChunks) || fUseWholeFile) && currentEntry < fNumEntries; currentChunk++) {

   //       // stop the loop when the loading is not active anymore
   //       {
   //          std::lock_guard<std::mutex> lock(fIsActiveMutex);
   //          if (!fIsActive)
   //             return;
   //       }

   //       // A pair that consists the proccessed, and passed events while loading the chunk
   //       std::size_t report = std::get<std::shared_ptr<RChunkLoader<Args...>>>(fChunkLoader)->LoadChunk(currentEntry);
   //       currentEntry += report;

   //       CreateBatches(report);
   //    }

   //    if (!fDropRemainder) {
   //       fBatchLoader->LastBatches();
   //    }

   //    fBatchLoader->DeActivate();
   // }


  void Sleep(unsigned int seconds) {
    usleep(seconds*1000);    
    std::cout << "Slept for " << seconds << " seconds" << std::endl;
  }
  
  TMVA::Experimental::RTensor<float> GenerateTrainBatch() {
    auto batchQueue = fBatchLoader->GetNumTrainingBatchQueue();
    std::cout << "Batches in queue: " << batchQueue << std::endl; 
    if (batchQueue < 6000) {
      std::cout << " " << std::endl;
      int Sleep1 = 1000;
      // std::thread th1([this, Sleep1](){ this->Sleep(Sleep1); });
      // std::thread th2([this, Sleep1](){ this->Sleep(Sleep1); });
      std::thread load([this]() { fChunkLoader->LoadTrainChunk(fTrainChunkTensor, fChunkNum); });
      load.join();
      // std::thread t1(RBatchGenerator::add, 1);
      // fChunkLoader->LoadTrainChunk(fTrainChunkTensor, fChunkNum);
      std::cout << "Loaded chunk: " << fChunkNum + 1 << std::endl;
      // std::cout << fTrainChunkTensor << std::endl;
      std::cout << " " << std::endl;
      std::thread loadBatch([this]() { fBatchLoader->CreateTrainingBatches(fTrainChunkTensor); });
      loadBatch.join();
      
      // fBatchLoader->CreateTrainingBatches(fTrainChunkTensor)
;          
      fChunkNum++;
    }
    // Get next batch if available
    return fBatchLoader->GetTrainBatch();
  }

  

  /// \brief Returns the next batch of validation data if available.
  /// Returns empty RTensor otherwise.
};
