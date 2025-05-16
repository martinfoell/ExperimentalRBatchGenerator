// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// #ifndef TMVA_RBATCHGENERATOR
// #define TMVA_RBATCHGENERATOR

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/BatchGenerator/RChunkLoader.hxx"
#include "TMVA/BatchGenerator/RBatchLoader.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>

#include "RChunkLoader.hxx"
#include "RBatchLoader.hxx"

// namespace TMVA {
// namespace Experimental {
// namespace Internal {

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
   std::size_t fTrainingChunkNum;
   std::size_t fValidationChunkNum;
   std::size_t fNumEpochs;
   std::size_t fCurrentEpoch;
   bool fShuffle;

   // ROOT::RDataFrame &f_rdf;
   ROOT::RDF::RNode &f_rdf;

   std::mutex fIsActiveMutex;
   bool fIsActive{false}; // Whether the loading thread is active
   bool fNotFiltered;

   bool fEpochActive{false};

   TMVA::Experimental::RTensor<float> fTrainTensor;
   TMVA::Experimental::RTensor<float> fTrainChunkTensor;

   TMVA::Experimental::RTensor<float> fValidationTensor;
   TMVA::Experimental::RTensor<float> fValidationChunkTensor;

   std::size_t fNumTrainingChunks;
   std::size_t fNumValidationChunks;

public:
   RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                   const std::size_t rangeSize, const std::size_t batchSize, const float validationSplit, bool shuffle,
                   const std::vector<std::string> &cols)
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

      fNumEntries = f_rdf.Count().GetValue();      
      fChunkLoader =
         std::make_unique<RChunkLoader<Args...>>(f_rdf, fNumEntries, fChunkSize, fRangeSize, fValidationSplit, fCols, fShuffle);
      fBatchLoader = std::make_unique<RBatchLoader>(fChunkSize, fBatchSize, fNumColumns);

      fChunkLoader->SplitDataset();

      fNumTrainingChunks = fChunkLoader->GetNumTrainChunks();
      fNumValidationChunks = fChunkLoader->GetNumValidationChunks();

      fCurrentEpoch = 0;

      fChunkNum = 0;
      fTrainingChunkNum = 0;
      fValidationChunkNum = 0;

      if (fNotFiltered) {
         std::cout << "Not filtered" << std::endl;
         // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksNoFilters, this);
      } else {
         std::cout << "Filtered: needs to be implemented" << std::endl;
         // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksFilters, this);
      }

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

   void ActivateEpoch() { fEpochActive = true; }

   void DeActivateEpoch() { fEpochActive = false; }

   TMVA::Experimental::RTensor<float> GetTrainBatch()
   {
      auto batchQueue = fBatchLoader->GetNumTrainingBatchQueue();

      // New epoch
      if (fEpochActive == false) {
         fChunkLoader->CreateTrainingChunksIntervals();
         fEpochActive = true;
         fTrainingChunkNum = 0;
      }

      if (batchQueue < 1 && fTrainingChunkNum < fNumTrainingChunks) {
         fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
         fBatchLoader->CreateTrainingBatches(fTrainChunkTensor);

         fTrainingChunkNum++;
      }
      // Get next batch if available
      return fBatchLoader->GetTrainBatch();
   }

   TMVA::Experimental::RTensor<float> GetValidationBatch()
   {
      auto batchQueue = fBatchLoader->GetNumValidationBatchQueue();

      // New epoch
      if (fEpochActive == false) {
         fChunkLoader->CreateValidationChunksIntervals();
         fEpochActive = true;
         fValidationChunkNum = 0;
      }

      if (batchQueue < 1 && fValidationChunkNum < fNumValidationChunks) {
         fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
         fBatchLoader->CreateValidationBatches(fValidationChunkTensor);
         fValidationChunkNum++;
      }

      // Get next batch if available
      return fBatchLoader->GetValidationBatch();
   }

   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
};

// } // namespace Internal
// } // namespace Experimental
// } // namespace TMVA

// #endif // TMVA_RBATCHGENERATOR
