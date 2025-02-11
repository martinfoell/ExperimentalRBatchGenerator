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
  std::size_t fNumEpochs;
  std::size_t fCurrentEpoch;  
  bool fShuffle;

    // ROOT::RDataFrame &f_rdf;
  ROOT::RDF::RNode &f_rdf;

  std::mutex fIsActiveMutex;
  bool fIsActive{false}; // Whether the loading thread is active
  bool fNotFiltered;

  bool fEpochActive{false};

  std::size_t fNumFullTrainChunks;
  std::size_t fReminderBatchSize;

  

  TMVA::Experimental::RTensor<float> fTrainTensor; 
  TMVA::Experimental::RTensor<float> fTrainChunkTensor;

  TMVA::Experimental::RTensor<float> fTrainBatchReminders;
  
  TMVA::Experimental::RTensor<float> fValidationTensor; 
  TMVA::Experimental::RTensor<float> fValidationChunkTensor;
  

 public:
  RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t numEpochs, const std::size_t chunkSize, const std::size_t rangeSize, const std::size_t batchSize,
                  const float validationSplit, bool shuffle, const std::vector<std::string> &cols)
    : f_rdf(rdf),
      fCols(cols),      
      fNumEpochs(numEpochs),
      fChunkSize(chunkSize),      
      fRangeSize(rangeSize),
      fBatchSize(batchSize),        
      fValidationSplit(validationSplit),
      fShuffle(shuffle),
      fNumColumns(cols.size()),
      fTrainTensor({0, 0}),
      fTrainChunkTensor({0, 0}),
      fTrainBatchReminders({0, 0}),            
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



    fReminderBatchSize = fChunkSize % fBatchSize;
    std::cout << "Reminder batch size: " << fReminderBatchSize << std::endl;
    fNumFullTrainChunks = fChunkLoader->GetNumberOfFullTrainingChunks();
    fTrainBatchReminders = fTrainBatchReminders.Resize({{fNumFullTrainChunks * fReminderBatchSize, fNumColumns}});            
    std::cout << "Number of Training chunks " << fNumFullTrainChunks << std::endl;
    
    fCurrentEpoch = 0;  

    fChunkNum = 0;    
  }

  ~RBatchGenerator() { DeActivate(); }


  std::size_t GetNumberOfTrainingChunks() {
    return fNumFullTrainChunks;
  }
  
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

  void ActivateEpoch() {
    fEpochActive = true;
  }

  void DeActivateEpoch() {
    fEpochActive = false;
  }

  
  
  TMVA::Experimental::RTensor<float> GetTrainBatch() {
    auto batchQueue = fBatchLoader->GetNumTrainingBatchQueue();
    // std::cout << "Batches in queue: " << batchQueue << std::endl; 
    
    if (fEpochActive == false) {
      std::cout << "------------ New Epoch ------------" << std::endl;
      fChunkLoader->CreateTrainRangeVector();     
      fEpochActive = true;
      fChunkNum = 0;
    }

    if (batchQueue < 1 && fChunkNum < fNumFullTrainChunks) {

      fChunkLoader->LoadTrainChunk(fTrainChunkTensor, fChunkNum);
      fBatchLoader->CreateTrainingBatches(fTrainChunkTensor);
      fBatchLoader->SaveReminderBatch(fTrainChunkTensor, fTrainBatchReminders, fChunkNum);
      
      // std::cout << " " << std::endl;      
      // std::cout << "Loaded chunk: " << fChunkNum + 1 << std::endl;
      // std::cout << fTrainChunkTensor << std::endl;
      // std::cout << " " << std::endl;

      // std::cout << "Saved reminder batch from chunk: " << std::endl;
      // std::cout << fTrainBatchReminders << std::endl;
      // std::cout << " " << std::endl;      
      // std::cout << "Reminder batch: " << *batch << std::endl;      

      
      fChunkNum++;
    }

    // else if (batchQueue < 1 && fChunkNum == fNumFullTrainChunks) {
    //   fChunkLoader->LoadTrainChunk(fTrainChunkTensor, fChunkNum);
    //   fBatchLoader->CreateTrainingBatches(fTrainChunkTensor);
    //   fBatchLoader->SaveReminderBatch(fTrainChunkTensor, fTrainBatchReminders, fChunkNum);
      
    //   std::cout << "Reminder batches here " << std::endl;
    //   fChunkNum++;
    // }
    // Get next batch if available
    return fBatchLoader->GetTrainBatch();
  }

  

  /// \brief Returns the next batch of validation data if available.
  /// Returns empty RTensor otherwise.
};
