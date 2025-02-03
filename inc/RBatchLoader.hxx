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
#include <queue>
#include <mutex>
#include <condition_variable>

class RBatchLoader {
private:
   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;
   std::size_t fTrainingRemainderRow = 0;
   std::size_t fValidationRemainderRow = 0;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;

public:
   RBatchLoader( std::size_t batchSize,
                std::size_t numColumns, std::size_t maxBatches)
      : fBatchSize(batchSize), fNumColumns(numColumns), fMaxBatches(maxBatches)
   {
      // Create remainders tensors
      fTrainingRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
      fValidationRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
   }

public:



   TMVA::Experimental::RTensor<float> GetTrainBatch()
   {
      // std::unique_lock<std::mutex> lock(fBatchLock);
      // fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();

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

   void CreateTrainingBatches(TMVA::Experimental::RTensor<float> &chunkTensor)
   {

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;
      
      for (std::size_t i = 0; i < 3; i++) {
         // Grab the first fBatchSize indices
         // Fill a batch
        batches.emplace_back(CreateBatch(chunkTensor, i));
      }
      
      for (std::size_t i = 0; i < batches.size(); i++) {
        fTrainingBatchQueue.push(std::move(batches[i]));
      }
      
   }

  // std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> GetTrainingBatchQueue() {
  //   return fTrainingBatchQueue;
  // }
  
  
};

