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

// #ifndef TMVA_RCHUNKLOADER
// #define TMVA_RCHUNKLOADER

#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

#include "RChunkConstructor.hxx"
// namespace TMVA {
// namespace Experimental {
// namespace Internal {

template <typename... ColTypes>
class RRangeChunkLoaderFunctor {
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   TMVA::Experimental::RTensor<float> &fChunkTensor;
   int fI;
   int fNumColumns;
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensorRange(const T &val, int i, int numColumns)
   {
      fChunkTensor.GetData()[fOffset++ + numColumns * i] = val;
   }

public:
   RRangeChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor, int i, int numColumns)
      : fChunkTensor(chunkTensor), fI(i), fNumColumns(numColumns)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 1;
      (AssignToTensorRange(cols, fI, fNumColumns), ...);
   }
};

template <typename... ColTypes>
class RChunkLoaderFunctorFilters {

private:
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   std::size_t fEntries{};
   std::size_t fChunkSize{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};

   TMVA::Experimental::RTensor<float> &fChunkTensor;
   TMVA::Experimental::RTensor<float> &fRemainderTensor;

   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), &fChunkTensor.GetData()[fOffset]);
         std::fill(&fChunkTensor.GetData()[fOffset + vec_size], &fChunkTensor.GetData()[fOffset + max_vec_size],
                   fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, &fChunkTensor.GetData()[fOffset]);
      }
      fOffset += max_vec_size;
      fEntries++;
   }

   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val)
   {
      fChunkTensor.GetData()[fOffset++] = val;
      fEntries++;
   }

public:
   RChunkLoaderFunctorFilters(TMVA::Experimental::RTensor<float> &chunkTensor,
                              TMVA::Experimental::RTensor<float> &remainderTensor, std::size_t entries,
                              std::size_t chunkSize, std::size_t &&offset,
                              const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                              const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor),
        fRemainderTensor(remainderTensor),
        fEntries(entries),
        fChunkSize(chunkSize),
        fOffset(offset),
        fMaxVecSizes(maxVecSizes),
        fVecPadding(vecPadding)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      if (fEntries == fChunkSize) {
         fChunkTensor = fRemainderTensor;
         fOffset = 0;
      }
      (AssignToTensor(cols), ...);
   }

   std::size_t &SetEntries() { return fEntries; }
   std::size_t &SetOffset() { return fOffset; }
};

template <typename... Args>
class RChunkLoader {
private:
   std::size_t fNumEntries;
   std::size_t fChunkSize;
   std::size_t fRangeSize;
   float fValidationSplit;

   std::size_t fNumTrainEntries;
   std::size_t fNumValidationEntries;

   // number of full training and validation chunks
   std::size_t fNumFullTrainChunks;
   std::size_t fNumFullValidationChunks;

   // number of reminder training and validation chunks
   std::size_t fNumReminderTrainChunks;
   std::size_t fNumReminderValidationChunks;

   // size of reminder training and validation chunks
   std::size_t fReminderTrainChunkSize;
   std::size_t fReminderValidationChunkSize;

   // total number of training and validation chunks
   std::size_t fNumTrainChunks;
   std::size_t fNumValidationChunks;

   // total number of full ranges, reminder ranges and total ranges in full chunks
   std::size_t fNumFullChunkFullRanges;
   std::size_t fNumFullChunkReminderRanges;
   std::size_t fNumFullChunkRanges;

   // total number of full ranges, reminder ranges and total ranges in full chunks
   std::size_t fNumReminderTrainChunkFullRanges;
   std::size_t fNumReminderValidationChunkFullRanges;

   std::size_t fNumReminderTrainChunkReminderRanges;
   std::size_t fNumReminderValidationChunkReminderRanges;

   std::size_t fNumReminderTrainChunkRanges;
   std::size_t fNumReminderValidationChunkRanges;

   std::size_t fFullChunkReminderRangeSize;
   std::size_t fReminderTrainChunkReminderRangeSize;
   std::size_t fReminderValidationChunkReminderRangeSize;

   std::size_t fTotNumFullChunks;
   std::size_t fTotNumReminderChunks;
   std::size_t fTotNumFullRanges;
   std::size_t fTotNumReminderRanges;
   std::size_t fTotNumReminderChunkFullRanges;

   std::size_t fTotEntriesFromRanges;

   std::size_t fTotNumTrainFullRanges;
   std::size_t fTotNumValidationFullRanges;

   std::size_t fTotNumTrainReminderRanges;
   std::size_t fTotNumValidationReminderRanges;

   std::vector<Long_t> fRangeVector;

   std::vector<Long_t> fPartialSumRangeSizes;

   std::vector<std::pair<Long64_t, Long64_t>> fFullRanges;
   std::vector<std::pair<Long64_t, Long64_t>> fReminderRanges;
   std::vector<std::pair<Long64_t, Long64_t>> fTrainRangesReminder;
   std::vector<std::pair<Long64_t, Long64_t>> fValidationRangesReminder;

   std::vector<std::pair<Long64_t, Long64_t>> fTrainRanges;
   std::vector<std::pair<Long64_t, Long64_t>> fValidationRanges;

   std::vector<std::pair<Long64_t, Long64_t>> fFullTrainRanges;
   std::vector<std::pair<Long64_t, Long64_t>> fReminderTrainRanges;

   std::vector<std::pair<Long64_t, Long64_t>> fFullValidationRanges;
   std::vector<std::pair<Long64_t, Long64_t>> fReminderValidationRanges;

   ROOT::RDF::RNode &f_rdf;
   // ROOT::RDataFrame &f_rdf;
   std::vector<std::string> fCols;
   std::size_t fNumCols;

   bool fNotFiltered;
   bool fShuffle;

   std::unique_ptr<RChunkConstructor> fTraining;
   std::unique_ptr<RChunkConstructor> fValidation;   

public:
   RChunkLoader(ROOT::RDF::RNode &rdf, const std::size_t chunkSize, const std::size_t rangeSize,
                const float validationSplit, const std::vector<std::string> &cols, bool shuffle)
      : f_rdf(rdf),
        fCols(cols),
        fChunkSize(chunkSize),
        fRangeSize(rangeSize),
        fValidationSplit(validationSplit),
        fNotFiltered(f_rdf.GetFilterNames().empty()),
        fShuffle(shuffle)

   {
      if (fNotFiltered) {
         fNumEntries = f_rdf.Count().GetValue();
      }

      fNumCols = fCols.size();

      // number of training and validation entries after the split
      fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
      fNumTrainEntries = fNumEntries - fNumValidationEntries;
      
      fTraining = std::make_unique<RChunkConstructor>(fNumTrainEntries, fChunkSize, fRangeSize);
      fValidation = std::make_unique<RChunkConstructor>(fNumValidationEntries, fChunkSize, fRangeSize);
      
      // number of full chunks for training and validetion
      fNumFullTrainChunks = fNumTrainEntries / fChunkSize;
      fNumFullValidationChunks = fNumValidationEntries / fChunkSize;

      // // total number of chunks from the dataset
      // fNumFullChunkRanges = fNumFullTrainChunks + fNumFullValidationChunks;

      // size of reminder chunk for training and validation
      fReminderTrainChunkSize = fNumTrainEntries % fChunkSize;
      fReminderValidationChunkSize = fNumValidationEntries % fChunkSize;

      // number of reminder chunks for training and validation (0 or 1)
      fNumReminderTrainChunks = fReminderTrainChunkSize == 0 ? 0 : 1;
      fNumReminderValidationChunks = fReminderValidationChunkSize == 0 ? 0 : 1;

      // total number of chunks for training and validation
      fNumTrainChunks = fNumFullTrainChunks + fNumReminderTrainChunks;
      fNumValidationChunks = fNumFullValidationChunks + fNumReminderValidationChunks;

      // number fo full ranges in a full chunk
      fNumFullChunkFullRanges = fChunkSize / fRangeSize;

      // size of the reminder range in a full chunk
      fFullChunkReminderRangeSize = fChunkSize % fRangeSize;

      // number of reminder ranges in a full chunk (0 or 1)
      fNumFullChunkReminderRanges = fFullChunkReminderRangeSize == 0 ? 0 : 1;

      // total number of ranges in a full chunk
      fNumFullChunkRanges = fNumFullChunkFullRanges + fNumFullChunkReminderRanges;

      // size of the reminder range in the reminder chunk for training and validation
      fReminderTrainChunkReminderRangeSize = fReminderTrainChunkSize % fRangeSize;
      fReminderValidationChunkReminderRangeSize = fReminderValidationChunkSize % fRangeSize;

      // number of full ranges in the reminder chunk for training and validation
      fNumReminderTrainChunkFullRanges = fReminderTrainChunkSize / fRangeSize;
      fNumReminderValidationChunkFullRanges = fReminderValidationChunkSize / fRangeSize;
      ;

      // number of reminder ranges in the reminder chunk for training and validation (0 or 1)
      fNumReminderTrainChunkReminderRanges = fReminderTrainChunkReminderRangeSize == 0 ? 0 : 1;
      fNumReminderValidationChunkReminderRanges = fReminderValidationChunkReminderRangeSize == 0 ? 0 : 1;

      // total number of ranges in the reminder chunk for training and validation
      fNumReminderTrainChunkRanges = fNumReminderTrainChunkFullRanges + fNumReminderTrainChunkReminderRanges;
      fNumReminderValidationChunkRanges =
         fNumReminderValidationChunkFullRanges + fNumReminderValidationChunkReminderRanges;

      // total number of full and reminder chunks in the dataset (train + val)
      fTotNumFullChunks = fNumFullTrainChunks + fNumFullValidationChunks;
      fTotNumReminderChunks = fNumReminderTrainChunks + fNumReminderValidationChunks;

      // total number of reminder  and reminder chunks in the dataset (train + val)
      fTotNumReminderChunkFullRanges = fNumReminderTrainChunkFullRanges + fNumReminderValidationChunkFullRanges;

      fTotNumTrainFullRanges = fNumFullTrainChunks * fNumFullChunkFullRanges + fNumReminderTrainChunkFullRanges;
      fTotNumValidationFullRanges =
         fNumFullValidationChunks * fNumFullChunkFullRanges + fNumReminderValidationChunkFullRanges;

      fTotNumTrainReminderRanges = fNumFullTrainChunks * fNumFullChunkReminderRanges;
      fTotNumValidationReminderRanges = fNumFullValidationChunks * fNumFullChunkReminderRanges;

      fTotNumFullRanges = fTotNumTrainFullRanges + fTotNumValidationFullRanges;
      fTotNumReminderRanges = fTotNumTrainReminderRanges + fTotNumValidationReminderRanges;

      fTotEntriesFromRanges = fTotNumFullRanges * fRangeSize + fTotNumReminderRanges * fFullChunkReminderRangeSize +
                              fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize +
                              fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize;
   }

   void PrintVector(std::vector<Long_t> vec) {
      std::cout << "{" ;
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1 ) {
            std::cout << vec[i];            
         }
         else {
            std::cout << vec[i] << ",";            
         }
      }
      std::cout << "}" << std::endl;
   }


   void PrintVectorSize(std::vector<std::size_t> vec) {
      std::cout << "{" ;
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1 ) {
            std::cout << vec[i];            
         }
         else {
            std::cout << vec[i] << ",";            
         }
      }
      std::cout << "}" << std::endl;
   }
   
   void PrintPair(std::vector<std::pair<Long_t, Long_t>> vec) {
      std::cout << "{" ;
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1 ) {
            std::cout << "(" << vec[i].first << ", " << vec[i].second << ")" ;            
         }
         else {
            std::cout << "(" << vec[i].first << ", " << vec[i].second << ")" << ","; 
            // std::cout << vec[i] << ",";            
         }
      }
      std::cout << "}" << std::endl;
   }

   void PrintChunk(std::vector<std::vector<std::pair<Long_t, Long_t>>> vec, int chunkNum) {
      std::vector<std::pair<Long_t, Long_t>> chunk = vec[chunkNum];

      PrintPair(chunk);
   }
   
   void SplitDataset()
   {
      // std::random_device rd;
      // std::mt19937 g(rd());
            
      std::mt19937 g(42);

      std::vector<Long_t> BlockSizes = {};
      // std::vector<Long_t> BlockBoundaries = {};
      
      // fill the training and validation block sizes
      for (size_t i = 0; i < fTraining->NumberOfDifferentBlocks.size(); i++) {
         BlockSizes.insert(BlockSizes.end(), fTraining->NumberOfDifferentBlocks[i], fTraining->SizeOfBlocks[i]);         
      }
      
      PrintVector(BlockSizes);
      for (size_t i = 0; i < fValidation->NumberOfDifferentBlocks.size(); i++) {
         BlockSizes.insert(BlockSizes.end(), fValidation->NumberOfDifferentBlocks[i], fValidation->SizeOfBlocks[i]);         
      }

      std::vector<Long_t> indices(BlockSizes.size());

      for (int i = 0; i < indices.size(); ++i) {
         indices[i] = i;
      }
   
      std::shuffle(indices.begin(), indices.end(), g);

      std::vector<Long_t> PermutedBlockSizes(BlockSizes.size());
      for (int i = 0; i < BlockSizes.size(); ++i) {
         PermutedBlockSizes[i] = BlockSizes[indices[i]];
      }

      std::vector<Long_t> BlockBoundaries(BlockSizes.size());      

      std::partial_sum(PermutedBlockSizes.begin(), PermutedBlockSizes.end(), BlockBoundaries.begin());      
      BlockBoundaries.insert(BlockBoundaries.begin(), 0);            

      std::vector<std::pair<Long_t, Long_t>> BlockIntervals;
      for (size_t i = 0; i < BlockBoundaries.size() - 1; ++i) {
        BlockIntervals.emplace_back(BlockBoundaries[i], BlockBoundaries[i + 1]);
      }

      // std::vector<Long_t> BlockBoundaries = {};
      std::vector<std::pair<Long_t, Long_t>> UnpermutedBlockIntervals(BlockIntervals.size());
      // std::vector<int> unshuffled(data.size());
      for (int i = 0; i < BlockIntervals.size(); ++i) {
         UnpermutedBlockIntervals[indices[i]] = BlockIntervals[i];
      }

      fTraining->BlockIntervals.insert(fTraining->BlockIntervals.begin(), UnpermutedBlockIntervals.begin(), UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks);
      fValidation->BlockIntervals.insert(fValidation->BlockIntervals.begin(), UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks, UnpermutedBlockIntervals.end());

      // PrintVector(indices) ;     
      // PrintVector(BlockSizes);
      // PrintVector(PermutedBlockSizes);
      // PrintVector(BlockBoundaries);
      // PrintPair(BlockIntervals);
      PrintPair(UnpermutedBlockIntervals);

      fTraining->DistributeBlockIntervals();
      fValidation->DistributeBlockIntervals();      
      
      std::cout << "========================= Before shiuffling =============================" << std::endl;      
      // std::cout << "Validation" << std::endl;
      // PrintVectorSize(fValidation->NumberOfDifferentBlocks);
      // PrintPair(fValidation->FullBlockIntervalsInFullChunks);
      // PrintPair(fValidation->LeftoverBlockIntervalsInFullChunks);      
      // PrintPair(fValidation->FullBlockIntervalsInLeftoverChunks);
      // PrintPair(fValidation->LeftoverBlockIntervalsInLeftoverChunks);      
   
      std::cout << std::endl;
      std::cout << "Training" << std::endl;
      std::cout << "Number of differnt blocks : ";
      PrintVectorSize(fTraining->NumberOfDifferentBlocks);
      std::cout << "Full blocks in full chunks: ";
      PrintPair(fTraining->FullBlockIntervalsInFullChunks);
      std::cout << "Leftover blocks in full chunks: ";      
      PrintPair(fTraining->LeftoverBlockIntervalsInFullChunks);      
      std::cout << "Full blocks in leftover chunks: ";      
      PrintPair(fTraining->FullBlockIntervalsInLeftoverChunks);
      std::cout << "Leftover blocks in leftover chunks: ";      
      PrintPair(fTraining->LeftoverBlockIntervalsInLeftoverChunks);      
   
      std::cout << std::endl;
      std::cout << "Validation" << std::endl;
      std::cout << "Number of differnt blocks : ";
      PrintVectorSize(fValidation->NumberOfDifferentBlocks);
      std::cout << "Full blocks in full chunks: ";
      PrintPair(fValidation->FullBlockIntervalsInFullChunks);
      std::cout << "Leftover blocks in full chunks: ";      
      PrintPair(fValidation->LeftoverBlockIntervalsInFullChunks);      
      std::cout << "Full blocks in leftover chunks: ";      
      PrintPair(fValidation->FullBlockIntervalsInLeftoverChunks);
      std::cout << "Leftover blocks in leftover chunks: ";      
      PrintPair(fValidation->LeftoverBlockIntervalsInLeftoverChunks);      
   
   }

   void CreateTrainingChunksIntervals() {
      
      std::mt19937 g(42);
      
      
      std::shuffle(fTraining->FullBlockIntervalsInFullChunks.begin(), fTraining->FullBlockIntervalsInFullChunks.end(), g);
      std::shuffle(fTraining->LeftoverBlockIntervalsInFullChunks.begin() ,fTraining->LeftoverBlockIntervalsInFullChunks.end(), g);      
      std::shuffle(fTraining->FullBlockIntervalsInLeftoverChunks.begin() ,fTraining->FullBlockIntervalsInLeftoverChunks.end(), g);
      std::shuffle(fTraining->LeftoverBlockIntervalsInLeftoverChunks.begin() ,fTraining->LeftoverBlockIntervalsInLeftoverChunks.end(), g);      

      fTraining->ChunksIntervals = {};
      fTraining->CreateChunksIntervals();

      std::shuffle(fTraining->ChunksIntervals.begin() ,fTraining->ChunksIntervals.end(), g);
      
      fTraining->SizeOfChunks();
      
      std::cout << std::endl;
      for (int i = 0; i < fTraining->Chunks; i++) {
         std::cout << "chunk " << i + 1 << ": ";
         PrintChunk(fTraining->ChunksIntervals, i);
      }

      std::cout << std::endl;
      std::cout << "Chunk sizes" << std::endl;
      PrintVectorSize(fTraining->ChunksSizes);
      std::cout << std::endl;      
   }

   void CreateValidationChunksIntervals() {
      
      std::mt19937 g(42);
      
      
      std::shuffle(fValidation->FullBlockIntervalsInFullChunks.begin(), fValidation->FullBlockIntervalsInFullChunks.end(), g);
      std::shuffle(fValidation->LeftoverBlockIntervalsInFullChunks.begin() ,fValidation->LeftoverBlockIntervalsInFullChunks.end(), g);      
      std::shuffle(fValidation->FullBlockIntervalsInLeftoverChunks.begin() ,fValidation->FullBlockIntervalsInLeftoverChunks.end(), g);
      std::shuffle(fValidation->LeftoverBlockIntervalsInLeftoverChunks.begin() ,fValidation->LeftoverBlockIntervalsInLeftoverChunks.end(), g);      

      fValidation->ChunksIntervals = {};      
      fValidation->CreateChunksIntervals();

      // std::shuffle(fValidation->ChunksIntervals.begin() ,fValidation->ChunksIntervals.end(), g);

      fValidation->SizeOfChunks();
      
      std::cout << std::endl;
      for (int i = 0; i < fValidation->Chunks; i++) {
         std::cout << "chunk " << i + 1 << ": ";
         PrintChunk(fValidation->ChunksIntervals, i);
      }

      std::cout << std::endl;
      std::cout << "Chunk sizes" << std::endl;
      PrintVectorSize(fValidation->ChunksSizes);
      std::cout << std::endl;      
      
   }
   

   void PrintChunks() {
      std::cout << "Training: ";      
      PrintPair(fTraining->BlockIntervals);
      std::cout << "Validation: ";      
      PrintPair(fValidation->BlockIntervals) ;     
      
      std::cout << std::endl;
      std::cout << "Training" << std::endl;
      std::cout << "Number of differnt blocks : ";
      PrintVectorSize(fTraining->NumberOfDifferentBlocks);
      std::cout << "Full blocks in full chunks: ";
      PrintPair(fTraining->FullBlockIntervalsInFullChunks);
      std::cout << "Leftover blocks in full chunks: ";     
      PrintPair(fTraining->LeftoverBlockIntervalsInFullChunks);      
      std::cout << "Full blocks in leftover chunks: ";      
      PrintPair(fTraining->FullBlockIntervalsInLeftoverChunks);
      std::cout << "Leftover blocks in leftover chunks: ";      
      PrintPair(fTraining->LeftoverBlockIntervalsInLeftoverChunks);      

      std::cout << std::endl;
      for (int i = 0; i < fTraining->Chunks; i++) {
         std::cout << "chunk " << i + 1 << ": ";
         PrintChunk(fTraining->ChunksIntervals, i);
      }


      std::cout << std::endl;
      for (int i = 0; i < fValidation->Chunks; i++) {
         std::cout << "chunk " << i + 1 << ": ";
         PrintChunk(fValidation->ChunksIntervals, i);
      }
            
   };
   
   void LoadTrainingChunkTest(TMVA::Experimental::RTensor<float> &TrainChunkTensor, std::size_t chunk)
   {

      std::random_device rd;
      std::mt19937 g(rd());

      std::size_t chunkSize = fTraining->ChunksSizes[chunk];
      
      if (chunk < fTraining->Chunks) {
         TMVA::Experimental::RTensor<float> Tensor({chunkSize, fNumCols});
         TrainChunkTensor = TrainChunkTensor.Resize({{chunkSize, fNumCols}});

         std::vector<int> indices(chunkSize);
         std::iota(indices.begin(), indices.end(), 0);


         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         std::vector<std::pair<Long_t, Long_t>> BlocksInChunk = fTraining->ChunksIntervals[chunk];
         for (std::size_t i = 0; i < BlocksInChunk.size(); i++) {

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
         }

         // shuffle data in RTensor
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      TrainChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   void LoadValidationChunkTest(TMVA::Experimental::RTensor<float> &ValidationChunkTensor, std::size_t chunk)
   {

      std::random_device rd;
      std::mt19937 g(rd());

      std::size_t chunkSize = fValidation->ChunksSizes[chunk];
      
      if (chunk < fValidation->Chunks) {
         TMVA::Experimental::RTensor<float> Tensor({chunkSize, fNumCols});
         ValidationChunkTensor = ValidationChunkTensor.Resize({{chunkSize, fNumCols}});

         std::vector<int> indices(chunkSize);
         std::iota(indices.begin(), indices.end(), 0);


         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         std::vector<std::pair<Long_t, Long_t>> BlocksInChunk = fValidation->ChunksIntervals[chunk];
         for (std::size_t i = 0; i < BlocksInChunk.size(); i++) {

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
         }

         // shuffle data in RTensor
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      ValidationChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   std::vector<std::size_t> GetTrainingChunkSizes() {return fTraining->ChunksSizes;}   
   std::vector<std::size_t> GetValidationChunkSizes() {return fValidation->ChunksSizes;}
  
   void CreateRangeVector()
   {
      std::random_device rd;
      std::mt19937 g(rd());

      std::vector<Long_t> RangeSizes = {};
      RangeSizes.insert(RangeSizes.end(), fTotNumFullRanges, fRangeSize);
      RangeSizes.insert(RangeSizes.end(), fTotNumReminderRanges, fFullChunkReminderRangeSize);
      RangeSizes.insert(RangeSizes.end(), fNumReminderTrainChunkReminderRanges, fReminderTrainChunkReminderRangeSize);
      RangeSizes.insert(RangeSizes.end(), fNumReminderValidationChunkReminderRanges,
                        fReminderValidationChunkReminderRangeSize);

      std::shuffle(RangeSizes.begin(), RangeSizes.end(), g);

      fPartialSumRangeSizes.resize(RangeSizes.size());

      std::partial_sum(RangeSizes.begin(), RangeSizes.end(), fPartialSumRangeSizes.begin());
      fPartialSumRangeSizes.insert(fPartialSumRangeSizes.begin(), 0);
   };

   void SortRangeVector()
   {

      std::random_device rd;
      std::mt19937 g(rd());

      for (size_t i = 0; i < fPartialSumRangeSizes.size() - 1; ++i) {
         int start = fPartialSumRangeSizes[i];
         int end = fPartialSumRangeSizes[i + 1];
         int rangeSize = end - start;

         if (rangeSize == fRangeSize) {
            fFullRanges.emplace_back(start, end); // Full-sized range
         } else if (rangeSize == fFullChunkReminderRangeSize) {
            fReminderRanges.emplace_back(start, end); // Reminder chunk
         } else if (rangeSize == fReminderTrainChunkReminderRangeSize) {
            fTrainRangesReminder.emplace_back(start, end); // Train reminder chunk
         } else if (rangeSize == fReminderValidationChunkReminderRangeSize) {
            fValidationRangesReminder.emplace_back(start, end); // Validation reminder chunk
         }
      }

      std::shuffle(fFullRanges.begin(), fFullRanges.end(), g);
      std::shuffle(fReminderRanges.begin(), fReminderRanges.end(), g);

      // bool allEqual = fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize &&
      //                 fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize &&
      //                 fFullChunkReminderRangeSize != 0;

      // bool fullEqualsTrain =
      //    fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize && fFullChunkReminderRangeSize != 0;

      // bool fullEqualsValidation =
      //    fFullChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize && fFullChunkReminderRangeSize !=
      //    0;

      // bool trainEqualsValidation = fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize
      // &&
      //                              fReminderTrainChunkReminderRangeSize != 0;

      // // i) All three reminder sizes are equal
      // if (allEqual) {
      //    if (!fReminderRanges.empty()) {
      //       fTrainRangesReminder.push_back(fReminderRanges.back());
      //       fReminderRanges.pop_back();
      //    }

      //    if (!fReminderRanges.empty()) {
      //       fValidationRangesReminder.push_back(fReminderRanges.back());
      //       fReminderRanges.pop_back();
      //    }

      //    std::cout << "i) Reminder range, reminder train range, and reminder validation range are equal\n";
      // }
      // // ii) Reminder and train sizes are equal
      // else if (fullEqualsTrain) {
      //    if (!fReminderRanges.empty()) {
      //       fTrainRangesReminder.push_back(fReminderRanges.back());
      //       fReminderRanges.pop_back();
      //    }

      //    std::cout << "ii) Reminder range and reminder train range are equal\n";
      // }
      // // iii) Reminder and validation sizes are equal
      // else if (fullEqualsValidation) {
      //    if (!fReminderRanges.empty()) {
      //       fValidationRangesReminder.push_back(fReminderRanges.back());
      //       fReminderRanges.pop_back();
      //    }

      //    std::cout << "iii) Reminder range and reminder validation range are equal\n";
      // }
      // // iv) Train and validation sizes are equal
      // else if (trainEqualsValidation) {
      //    if (!fTrainRangesReminder.empty()) {
      //       fValidationRangesReminder.push_back(fTrainRangesReminder.back());
      //       fTrainRangesReminder.pop_back();
      //    }

      //    std::cout << "iv) Reminder train range and reminder validation range are equal\n";
      // }

      // corner cases if some of the reminder sizes are equal
      if (fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
          fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
          fFullChunkReminderRangeSize != 0) {
         fTrainRangesReminder.push_back(fReminderRanges.back());
         fReminderRanges.pop_back();

         fValidationRangesReminder.push_back(fReminderRanges.back());
         fReminderRanges.pop_back();

         std::cout << "i) Reminder range, reminder train range and reminder validation range are equal " << std::endl;
      }

      else if (fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
               fFullChunkReminderRangeSize != 0) {
         fTrainRangesReminder.push_back(fReminderRanges.back());
         fReminderRanges.pop_back();

         std::cout << "ii) Reminder range and reminder train range are equal " << std::endl;
      }

      else if (fFullChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
               fFullChunkReminderRangeSize != 0) {
         fValidationRangesReminder.push_back(fReminderRanges.back());
         fReminderRanges.pop_back();

         std::cout << "iii) Reminder range and reminder validation range are equal " << std::endl;
      }

      else if (fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
               fReminderTrainChunkReminderRangeSize != 0) {
         fValidationRangesReminder.push_back(fTrainRangesReminder.back());
         fTrainRangesReminder.pop_back();

         std::cout << "iv) Reminder train range and reminder validation range are equal " << std::endl;
      }

      // change to span later
      fFullTrainRanges =
         std::vector<std::pair<Long64_t, Long64_t>>(fFullRanges.begin(), fFullRanges.begin() + fTotNumTrainFullRanges);
      fReminderTrainRanges = std::vector<std::pair<Long64_t, Long64_t>>(
         fReminderRanges.begin(), fReminderRanges.begin() + fTotNumTrainReminderRanges);

      fFullValidationRanges = std::vector<std::pair<Long64_t, Long64_t>>(fFullRanges.begin() + fTotNumTrainFullRanges,
                                                                         fFullRanges.begin() + fTotNumTrainFullRanges +
                                                                            fTotNumValidationFullRanges);
      fReminderValidationRanges = std::vector<std::pair<Long64_t, Long64_t>>(
         fReminderRanges.begin() + fTotNumTrainReminderRanges,
         fReminderRanges.begin() + fTotNumTrainReminderRanges + fTotNumValidationReminderRanges);
   };

   void CreateTrainRangeVector()
   {

      std::random_device rd;
      std::mt19937 g(rd());

      fTrainRanges = {};

      std::shuffle(fFullTrainRanges.begin(), fFullTrainRanges.end(), g);
      std::shuffle(fReminderTrainRanges.begin(), fReminderTrainRanges.end(), g);

      // Fill full chunks
      for (std::size_t i = 0; i < fNumFullTrainChunks; i++) {
         std::size_t startFull = i * fNumFullChunkFullRanges;
         std::size_t endFull = (i + 1) * fNumFullChunkFullRanges;

         std::size_t startReminder = i * fNumFullChunkReminderRanges;
         std::size_t endReminder = (i + 1) * fNumFullChunkReminderRanges;

         std::move(fFullTrainRanges.begin() + startFull, fFullTrainRanges.begin() + endFull,
                   std::back_inserter(fTrainRanges));
         std::move(fReminderTrainRanges.begin() + startReminder, fReminderTrainRanges.begin() + endReminder,
                   std::back_inserter(fTrainRanges));
      }

      // Fill reminder chunks
      std::size_t startFullReminder = fNumFullTrainChunks * fNumFullChunkFullRanges;
      std::size_t endFullReminder = startFullReminder + fNumReminderTrainChunkFullRanges;

      std::size_t endReminderReminder = fNumReminderTrainChunkReminderRanges;
      std::move(fFullTrainRanges.begin() + startFullReminder, fFullTrainRanges.begin() + endFullReminder,
                std::back_inserter(fTrainRanges));
      std::move(fTrainRangesReminder.begin(), fTrainRangesReminder.begin() + endReminderReminder,
                std::back_inserter(fTrainRanges));
   }

   void CreateValidationRangeVector()
   {

      std::random_device rd;
      std::mt19937 g(rd());

      fValidationRanges = {};

      std::shuffle(fFullValidationRanges.begin(), fFullValidationRanges.end(), g);
      std::shuffle(fReminderValidationRanges.begin(), fReminderValidationRanges.end(), g);

      // Fill full chunks
      for (std::size_t i = 0; i < fNumFullValidationChunks; i++) {
         std::size_t startFull = i * fNumFullChunkFullRanges;
         std::size_t endFull = (i + 1) * fNumFullChunkFullRanges;

         std::size_t startReminder = i * fNumFullChunkReminderRanges;
         std::size_t endReminder = (i + 1) * fNumFullChunkReminderRanges;
         std::move(fFullValidationRanges.begin() + startFull, fFullValidationRanges.begin() + endFull,
                   std::back_inserter(fValidationRanges));
         std::move(fReminderValidationRanges.begin() + startReminder, fReminderValidationRanges.begin() + endReminder,
                   std::back_inserter(fValidationRanges));
      }

      // Fill reminder chunk
      std::size_t startFullReminder = fNumFullValidationChunks * fNumFullChunkFullRanges;
      std::size_t endFullReminder = startFullReminder + fNumReminderValidationChunkFullRanges;

      std::size_t endReminderReminder = fNumReminderValidationChunkReminderRanges;
      std::move(fFullValidationRanges.begin() + startFullReminder, fFullValidationRanges.begin() + endFullReminder,
                std::back_inserter(fValidationRanges));
      std::move(fValidationRangesReminder.begin(), fValidationRangesReminder.begin() + endReminderReminder,
                std::back_inserter(fValidationRanges));
   }

   void Start()
   {
      CreateRangeVector();
      SortRangeVector();
   }

   void LoadTrainingDataset(TMVA::Experimental::RTensor<float> &TrainTensor)
   {
      TMVA::Experimental::RTensor<float> Tensor({fNumTrainEntries, fNumCols});
      TrainTensor = TrainTensor.Resize({{fNumTrainEntries, fNumCols}});

      std::vector<int> indices(fNumTrainEntries);
      std::iota(indices.begin(), indices.end(), 0);

      std::random_device rd;
      std::mt19937 g(rd());

      if (fShuffle) {
         std::shuffle(indices.begin(), indices.end(), g);
      }

      int chunkEntry = 0;
      for (int i = 0; i < fTrainRanges.size(); i++) {
         RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fTrainRanges[i].first, fTrainRanges[i].second);
         f_rdf.Foreach(func, fCols);
         chunkEntry += fTrainRanges[i].second - fTrainRanges[i].first;
      }

      for (int i = 0; i < fNumTrainEntries; i++) {
         std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                   TrainTensor.GetData() + i * fNumCols);
      }
   }

   void LoadValidationDataset(TMVA::Experimental::RTensor<float> &ValidationTensor)
   {
      TMVA::Experimental::RTensor<float> Tensor({fNumValidationEntries, fNumCols});
      ValidationTensor = ValidationTensor.Resize({{fNumValidationEntries, fNumCols}});

      std::vector<int> indices(fNumValidationEntries);
      std::iota(indices.begin(), indices.end(), 0);

      std::random_device rd;
      std::mt19937 g(rd());

      if (fShuffle) {
         std::shuffle(indices.begin(), indices.end(), g);
      }

      int chunkEntry = 0;
      for (int i = 0; i < fValidationRanges.size(); i++) {
         RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fValidationRanges[i].first, fValidationRanges[i].second);
         f_rdf.Foreach(func, fCols);
         chunkEntry += fValidationRanges[i].second - fValidationRanges[i].first;
      }

      for (int i = 0; i < fNumValidationEntries; i++) {
         std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                   ValidationTensor.GetData() + i * fNumCols);
      }
   }

   void LoadTrainChunk(TMVA::Experimental::RTensor<float> &TrainChunkTensor, std::size_t chunk)
   {

      if (chunk < fNumFullTrainChunks) {
         TMVA::Experimental::RTensor<float> Tensor({fChunkSize, fNumCols});
         TrainChunkTensor = TrainChunkTensor.Resize({{fChunkSize, fNumCols}});

         std::vector<int> indices(fChunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         std::random_device rd;
         std::mt19937 g(rd());

         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         for (std::size_t i = 0; i < fNumFullChunkRanges; i++) {
            std::size_t entry = chunk * fNumFullChunkRanges + i;

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fTrainRanges[entry].first, fTrainRanges[entry].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += fTrainRanges[entry].second - fTrainRanges[entry].first;
         }

         for (std::size_t i = 0; i < fChunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      TrainChunkTensor.GetData() + i * fNumCols);
         }
      }

      else {
         TMVA::Experimental::RTensor<float> Tensor({fReminderTrainChunkSize, fNumCols});
         TrainChunkTensor = TrainChunkTensor.Resize({{fReminderTrainChunkSize, fNumCols}});

         std::vector<int> indices(fReminderTrainChunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         std::random_device rd;
         std::mt19937 g(rd());

         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         for (std::size_t i = 0; i < fNumReminderTrainChunkRanges; i++) {
            std::size_t entry = chunk * fNumFullChunkRanges + i;

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fTrainRanges[entry].first, fTrainRanges[entry].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += fTrainRanges[entry].second - fTrainRanges[entry].first;
         }

         for (std::size_t i = 0; i < fReminderTrainChunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      TrainChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   void LoadValidationChunk(TMVA::Experimental::RTensor<float> &ValidationChunkTensor, std::size_t chunk)
   {

      if (chunk < fNumFullValidationChunks) {
         TMVA::Experimental::RTensor<float> Tensor({fChunkSize, fNumCols});
         ValidationChunkTensor = ValidationChunkTensor.Resize({{fChunkSize, fNumCols}});

         std::vector<int> indices(fChunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         std::random_device rd;
         std::mt19937 g(rd());

         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         for (std::size_t i = 0; i < fNumFullChunkRanges; i++) {
            std::size_t entry = chunk * fNumFullChunkRanges + i;

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fValidationRanges[entry].first,
                                                          fValidationRanges[entry].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += fValidationRanges[entry].second - fValidationRanges[entry].first;
         }

         for (std::size_t i = 0; i < fChunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      ValidationChunkTensor.GetData() + i * fNumCols);
         }
      }

      else {
         TMVA::Experimental::RTensor<float> Tensor({fReminderValidationChunkSize, fNumCols});
         ValidationChunkTensor = ValidationChunkTensor.Resize({{fReminderValidationChunkSize, fNumCols}});

         std::vector<int> indices(fReminderValidationChunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         std::random_device rd;
         std::mt19937 g(rd());

         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         for (std::size_t i = 0; i < fNumReminderValidationChunkRanges; i++) {
            std::size_t entry = chunk * fNumFullChunkRanges + i;

            RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fValidationRanges[entry].first,
                                                          fValidationRanges[entry].second);
            f_rdf.Foreach(func, fCols);
            chunkEntry += fValidationRanges[entry].second - fValidationRanges[entry].first;
         }

         for (std::size_t i = 0; i < fReminderValidationChunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      ValidationChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   void CheckIfUnique(TMVA::Experimental::RTensor<float> &Tensor)
   {
      auto tensorSize = Tensor.GetSize();
      TMVA::Experimental::RTensor<float> SqueezeTensor = Tensor.Reshape({1, tensorSize}).Squeeze();

      std::list<int> allEntries;
      for (int i = 0; i < tensorSize; i++) {
         allEntries.push_back(SqueezeTensor(0, i));
      }
      allEntries.sort();
      allEntries.unique();
      if (allEntries.size() == tensorSize) {
         std::cout << "Tensor consists of only unique elements" << std::endl;
      }
   };

   void CheckIfOverlap(TMVA::Experimental::RTensor<float> &Tensor1, TMVA::Experimental::RTensor<float> &Tensor2)
   {
      auto tensorSize1 = Tensor1.GetSize();
      TMVA::Experimental::RTensor<float> SqueezeTensor1 = Tensor1.Reshape({1, tensorSize1}).Squeeze();

      std::list<int> allEntries1;
      for (int i = 0; i < tensorSize1; i++) {
         allEntries1.push_back(SqueezeTensor1(0, i));
      }

      auto tensorSize2 = Tensor2.GetSize();
      TMVA::Experimental::RTensor<float> SqueezeTensor2 = Tensor2.Reshape({1, tensorSize2}).Squeeze();

      std::list<int> allEntries2;
      for (int i = 0; i < tensorSize2; i++) {
         allEntries2.push_back(SqueezeTensor2(0, i));
      }

      std::set<int> result;

      // Call the set_intersection(), which computes the
      // intersection of set1 and set2 and
      // inserts the result into the 'result' set
      std::set<int> set1(allEntries1.begin(), allEntries1.end());
      std::set<int> set2(allEntries2.begin(), allEntries2.end());
      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), inserter(result, result.begin()));
      // std::list<int> result = intersection(allEntries1, allEntries2);

      if (result.size() == 0) {
         std::cout << "No overlap between the tensors" << std::endl;
      } else {
         std::cout << "Intersection between tensors: ";
         for (int num : result) {
            std::cout << num << " ";
         }
         std::cout << std::endl;
      }
   };

   std::size_t GetNumTrainChunks() { return fTraining->Chunks; }

   std::size_t GetNumValidationChunks() { return fValidation->Chunks; }

   std::size_t GetNumberOfFullTrainingChunks() { return fNumFullTrainChunks; }

   std::size_t GetNumberOfFullValidationChunks() { return fNumFullValidationChunks; }

   void PrintTrainValidationVector()
   {

      std::cout << " " << std::endl;
      std::cout << "Full ranges: " << fFullRanges.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fFullRanges) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;

      std::cout << "Reminder ranges: " << fReminderRanges.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fReminderRanges) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;

      std::cout << "Reminder Train ranges: " << fTrainRangesReminder.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fTrainRangesReminder) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;

      std::cout << "Reminder Validation ranges: " << fValidationRangesReminder.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fValidationRangesReminder) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;

      // CreateTrainValidationRangeVectors();

      std::cout << " " << std::endl;
      std::cout << "Train ranges: " << fTrainRanges.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fTrainRanges) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;

      std::cout << "Validation ranges: " << fValidationRanges.size() << std::endl;
      std::cout << "-------------------------" << std::endl;
      for (auto i : fValidationRanges) {
         std::cout << "(" << i.first << ", " << i.second << ")"
                   << ", ";
      }
      std::cout << " " << std::endl;
      std::cout << "-------------------------" << std::endl;
      std::cout << " " << std::endl;
   }

   void PrintRangeVector()
   {
      std::cout << "{";
      for (auto i : fRangeVector) {
         std::cout << i << ", ";
      }
      std::cout << "}" << std::endl;
   }

   // to do: make variables which count how many ranges of each sort

   void PrintRow(std::string title, int col1, int col2, int col3, int col4, int colWidthS, int colWidth)
   {
      std::cout << std::left;
      std::cout << std::setw(colWidthS) << title << std::setw(colWidth) << col1 << std::setw(colWidth) << col2
                << std::setw(colWidth) << col3 << std::setw(colWidth) << col4 << std::endl;
   };

   void PrintRowB(std::string title, int col1, int col2, int col3, int col4, int col5, int colWidthS, int colWidth)
   {
      std::cout << std::left;
      std::cout << std::setw(colWidthS) << title << std::setw(colWidth) << col1 << std::setw(colWidth) << col2
                << std::setw(colWidth) << col3 << std::setw(colWidth) << col4 << std::setw(colWidth) << col5
                << std::endl;
   };

   void
   PrintRowHeader(std::string title, string col1, string col2, string col3, string col4, int colWidthS, int colWidth)
   {
      std::cout << std::string(colWidthS + 4 * colWidth, '=') << std::endl;
      std::cout << std::left;
      std::cout << std::setw(colWidthS) << title << std::setw(colWidth) << col1 << std::setw(colWidth) << col2
                << std::setw(colWidth) << col3 << std::setw(colWidth) << col4 << std::endl;
      std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
   };

   void PrintRowHeaderB(std::string title, string col1, string col2, string col3, string col4, string col5,
                        int colWidthS, int colWidth)
   {
      std::cout << std::string(colWidthS + 5 * colWidth, '=') << std::endl;
      std::cout << std::left;
      std::cout << std::setw(colWidthS) << title << std::setw(colWidth) << col1 << std::setw(colWidth) << col2
                << std::setw(colWidth) << col3 << std::setw(colWidth) << col4 << std::setw(colWidth) << col5
                << std::endl;
      std::cout << std::string(colWidthS + 5 * colWidth, '-') << std::endl;
   };

   void PrintChunkDistributions()
   {

      const int colWidthS = 25;
      const int colWidth = 10;

      // std::cout << std::left;
      std::cout << std::setw(colWidthS + colWidth) << "Train" << std::setw(2 * colWidth) << "Validation" << std::endl;

      PrintRowHeader("Entries", "Number", "Size", "Number", "Size", colWidthS, colWidth);
      PrintRow("Total", 1, fNumTrainEntries, 1, fNumValidationEntries, colWidthS, colWidth);
      std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
      std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;

      PrintRowHeader("Chunk distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);
      PrintRow("Full", fNumFullTrainChunks, fChunkSize, fNumFullValidationChunks, fChunkSize, colWidthS, colWidth);
      PrintRow("Reminder", fNumReminderTrainChunks, fReminderTrainChunkSize, fNumReminderValidationChunks,
               fReminderValidationChunkSize, colWidthS, colWidth);
      std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
      std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;
   }

   void PrintRangeDistributions()
   {

      const int colWidthS = 25;
      const int colWidth = 10;

      std::cout << std::left;
      std::cout << "Full chunks" << std::endl;
      PrintRowHeader("Range distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);
      PrintRow("Full", fNumFullChunkFullRanges, fRangeSize, fNumFullChunkFullRanges, fRangeSize, colWidthS, colWidth);
      PrintRow("Reminder", fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, fNumFullChunkReminderRanges,
               fFullChunkReminderRangeSize, colWidthS, colWidth);
      std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
      std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;

      std::cout << std::left;
      std::cout << "Reminder chunks" << std::endl;
      PrintRowHeader("Range distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);
      PrintRow("Full", fNumReminderTrainChunkFullRanges, fRangeSize, fNumReminderValidationChunkFullRanges, fRangeSize,
               colWidthS, colWidth);
      PrintRow("Reminder", fNumReminderTrainChunkReminderRanges, fReminderTrainChunkReminderRangeSize,
               fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize, colWidthS,
               colWidth);
      std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
      std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;

      PrintRowHeaderB("Ranges", "Number", "Size", "Number", "Size", "Entries", colWidthS, colWidth);
      PrintRowB("Full", fTotNumTrainFullRanges, fRangeSize, fTotNumValidationFullRanges, fRangeSize,
                fTotNumTrainFullRanges * fRangeSize + fTotNumValidationFullRanges * fRangeSize, colWidthS, colWidth);
      PrintRowB("Reminder", fTotNumTrainReminderRanges, fFullChunkReminderRangeSize, fTotNumValidationReminderRanges,
                fFullChunkReminderRangeSize,
                fTotNumTrainReminderRanges * fFullChunkReminderRangeSize +
                   fTotNumValidationReminderRanges * fFullChunkReminderRangeSize,
                colWidthS, colWidth);
      PrintRowB("Reminder Train", fNumReminderTrainChunkReminderRanges, fReminderTrainChunkReminderRangeSize, 0, 0,
                fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize, colWidthS, colWidth);
      PrintRowB("Reminder Validation", 0, 0, fNumReminderValidationChunkReminderRanges,
                fReminderValidationChunkReminderRangeSize,
                fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize, colWidthS,
                colWidth);

      std::cout << std::string(colWidthS + 5 * colWidth, '-') << std::endl;
      PrintRowB("Total", fTotNumFullRanges * fRangeSize, fTotNumReminderRanges * fFullChunkReminderRangeSize,
                fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize,
                fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize,
                fTotEntriesFromRanges, colWidthS, colWidth);

      std::cout << fTotNumFullRanges << " " << fTotNumReminderRanges << std::endl;
   }
};

template <typename... Args>
class RChunkLoaderFilters {

private:
   ROOT::RDF::RNode &f_rdf;
   TMVA::Experimental::RTensor<float> &fChunkTensor;

   std::size_t fChunkSize;
   std::vector<std::string> fCols;
   const std::size_t fNumEntries;
   std::size_t fNumAllEntries;
   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumColumns;

   const std::size_t fPartOfChunkSize;
   TMVA::Experimental::RTensor<float> fRemainderChunkTensor;
   std::size_t fRemainderChunkTensorRow = 0;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param filters
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoaderFilters(ROOT::RDF::RNode &rdf, TMVA::Experimental::RTensor<float> &chunkTensor,
                       const std::size_t chunkSize, const std::vector<std::string> &cols, std::size_t numEntries,
                       std::size_t numAllEntries, const std::vector<std::size_t> &vecSizes = {},
                       const float vecPadding = 0.0)
      : f_rdf(rdf),
        fChunkTensor(chunkTensor),
        fChunkSize(chunkSize),
        fCols(cols),
        fNumEntries(numEntries),
        fNumAllEntries(numAllEntries),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size()),
        fPartOfChunkSize(chunkSize / 5),
        fRemainderChunkTensor(std::vector<std::size_t>{fPartOfChunkSize, fNumColumns})
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t, std::size_t> LoadChunk(std::size_t currentRow)
   {
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++) {
         std::copy(fRemainderChunkTensor.GetData() + (i * fNumColumns),
                   fRemainderChunkTensor.GetData() + ((i + 1) * fNumColumns),
                   fChunkTensor.GetData() + (i * fNumColumns));
      }

      RChunkLoaderFunctorFilters<Args...> func(fChunkTensor, fRemainderChunkTensor, fRemainderChunkTensorRow,
                                               fChunkSize, fRemainderChunkTensorRow * fNumColumns, fVecSizes,
                                               fVecPadding);

      std::size_t passedEvents = 0;
      std::size_t processedEvents = 0;

      while ((passedEvents < fChunkSize && passedEvents < fNumEntries) && currentRow < fNumAllEntries) {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fPartOfChunkSize);
         auto report = f_rdf.Report();

         f_rdf.Foreach(func, fCols);

         processedEvents += report.begin()->GetAll();
         passedEvents += (report.end() - 1)->GetPass();

         currentRow += fPartOfChunkSize;
         func.SetEntries() = passedEvents;
         func.SetOffset() = passedEvents * fNumColumns;
      }

      fRemainderChunkTensorRow = passedEvents > fChunkSize ? passedEvents - fChunkSize : 0;

      return std::make_pair(processedEvents, passedEvents);
   }

   std::size_t LastChunk()
   {
      for (std::size_t i = 0; i < fRemainderChunkTensorRow; i++) {
         std::copy(fRemainderChunkTensor.GetData() + (i * fNumColumns),
                   fRemainderChunkTensor.GetData() + ((i + 1) * fNumColumns),
                   fChunkTensor.GetData() + (i * fNumColumns));
      }

      return fRemainderChunkTensorRow;
   }
};

// } // namespace Internal
// } // namespace Experimental
// } // namespace TMVA
// #endif // TMVA_RCHUNKLOADER
