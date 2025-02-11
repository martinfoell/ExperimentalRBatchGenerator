#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>
#include <iostream>
#include <list>
#include <set>


// Imports for threading
// #include <queue>
// #include <mutex>
// #include <condition_variable>

class RBatchLoader {
 private:
  std::size_t fBatchSize;
  std::size_t fChunkSize;  
  std::size_t fNumColumns;
  std::size_t fMaxBatches;
  std::size_t fTrainingRemainderRow = 0;
  std::size_t fValidationRemainderRow = 0;


  std::size_t fNumChunkBatches;
  std::size_t fChunkReminderBatchSize;
  
  bool fIsActive = false;

  std::mutex fBatchLock;
  std::condition_variable fBatchCondition;

  std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
  std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;

  std::size_t fNumTrainingBatchQueue;
  std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

  std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
  std::unique_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;

 public:
  RBatchLoader( std::size_t chunkSize, std::size_t batchSize,
                std::size_t numColumns)
    : fChunkSize(chunkSize),
      fBatchSize(batchSize),
      fNumColumns(numColumns)
        
  {
    fNumTrainingBatchQueue = fTrainingBatchQueue.size();
    fNumChunkBatches = fChunkSize / fBatchSize;
    fChunkReminderBatchSize = fChunkSize % fBatchSize;
  }

 public:

  void Activate()
  {
    // fTrainingRemainderRow = 0;
    // fValidationRemainderRow = 0;

    {
      std::lock_guard<std::mutex> lock(fBatchLock);
      fIsActive = true;
    }
    fBatchCondition.notify_all();
  }

  /// \brief DeActivate the batchloader. This means that no more batches are created.
  /// Batches can still be returned if they are already loaded
  void DeActivate()
  {
    {
      std::lock_guard<std::mutex> lock(fBatchLock);
      fIsActive = false;
    }
    fBatchCondition.notify_all();
  }


  TMVA::Experimental::RTensor<float> GetTrainBatch()
  {

    if (fTrainingBatchQueue.empty()) {
      fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
      return *fCurrentBatch;
    }

    fCurrentBatch = std::move(fTrainingBatchQueue.front());
    fTrainingBatchQueue.pop();

    // std::cout << *fCurrentBatch << std::endl;
    return *fCurrentBatch;
  }
  
  /// \brief Return a batch of data as a unique pointer.
  /// After the batch has been processed, it should be destroyed.
  /// \return Training batch
  std::unique_ptr<TMVA::Experimental::RTensor<float>>
  CreateBatch(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t idxs) {
    auto batch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));
    std::copy(chunkTensor.GetData() + (idxs * fBatchSize * fNumColumns),
              chunkTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

    return batch;
  }


  void SaveReminderBatch(TMVA::Experimental::RTensor<float> &chunkTensor, TMVA::Experimental::RTensor<float> &reminderBatchesTensor, std::size_t idxs)
  {
    std::copy(chunkTensor.GetData() + (fNumChunkBatches * fBatchSize * fNumColumns),
              chunkTensor.GetData() + (fNumChunkBatches *  fBatchSize * fNumColumns + fChunkReminderBatchSize * fNumColumns),
              reminderBatchesTensor.GetData() + (idxs * fChunkReminderBatchSize * fNumColumns));
  }
  
  void CreateTrainingBatches(TMVA::Experimental::RTensor<float> &chunkTensor)
  {

    std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;
      
    std::cout << "Num batches in chunk " << fNumChunkBatches << std::endl; 
    for (std::size_t i = 0; i < fNumChunkBatches; i++) {
      // Fill a batch
      batches.emplace_back(CreateBatch(chunkTensor, i));
    }
      
    for (std::size_t i = 0; i < batches.size(); i++) {
      fTrainingBatchQueue.push(std::move(batches[i]));
    }
      
  }

  // void CopyReminderBatch()

  // std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> GetTrainingBatchQueue() {
  //   return fTrainingBatchQueue;
  // }
  
  std::size_t GetNumTrainingBatchQueue() {
    return fTrainingBatchQueue.size();
  }

};

