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

   ROOT::RDF::RNode &f_rdf;
   std::vector<std::string> fCols;
   std::size_t fNumCols;

   bool fNotFiltered;
   bool fShuffle;

   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;

   std::unique_ptr<RChunkConstructor> fTraining;
   std::unique_ptr<RChunkConstructor> fValidation;

public:
   RChunkLoader(ROOT::RDF::RNode &rdf, std::size_t numEntries, const std::size_t chunkSize, const std::size_t rangeSize,
                const float validationSplit, const std::vector<std::string> &cols, bool shuffle)
      : f_rdf(rdf),
        fNumEntries(numEntries),
        fCols(cols),
        fChunkSize(chunkSize),
        fRangeSize(rangeSize),
        fValidationSplit(validationSplit),
        fNotFiltered(f_rdf.GetFilterNames().empty()),
        fShuffle(shuffle)

   {
      // if (fNotFiltered) {
      //    fNumEntries = f_rdf.Count().GetValue();
      // }

      // fNumEntries = f_rdf.Count().GetValue();
      if (fNotFiltered) {
         // fNumEntries = f_rdf.Count().GetValue();
         std::cout << "Entries: " << fNumEntries << std::endl;
         std::cout << "Not filtered" << std::endl;
         // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksNoFilters, this);
      } else {
         auto report = f_rdf.Report();
         fEntries = f_rdf.Take<ULong64_t>("rdfentry_");
         // add the last element in entries to not go out range when filling chunk
         fEntries->push_back((*fEntries)[fNumEntries - 1] + 1);         
         for (int i = 0; i < fEntries->size(); i++) {
            std::cout << (*fEntries)[i] << std::endl;
         }
         std::cout << "Size of fEntries: " << fEntries->size() << std::endl;
         std::size_t numAllEntries = report.begin()->GetAll();
         
         std::cout << "Filtered: needs to be implemented" << std::endl;
         std::cout << "Entries: " << fNumEntries << " " << numAllEntries << std::endl;         
      }

      fNumCols = fCols.size();

      // number of training and validation entries after the split
      fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
      fNumTrainEntries = fNumEntries - fNumValidationEntries;

      fTraining = std::make_unique<RChunkConstructor>(fNumTrainEntries, fChunkSize, fRangeSize);
      fValidation = std::make_unique<RChunkConstructor>(fNumValidationEntries, fChunkSize, fRangeSize);
      
   }

   void PrintVector(std::vector<Long_t> vec)
   {
      std::cout << "{";
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1) {
            std::cout << vec[i];
         } else {
            std::cout << vec[i] << ",";
         }
      }
      std::cout << "}" << std::endl;
   }

   void PrintVectorSize(std::vector<std::size_t> vec)
   {
      std::cout << "{";
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1) {
            std::cout << vec[i];
         } else {
            std::cout << vec[i] << ",";
         }
      }
      std::cout << "}" << std::endl;
   }

   void PrintPair(std::vector<std::pair<Long_t, Long_t>> vec)
   {
      std::cout << "{";
      for (int i = 0; i < vec.size(); i++) {
         if (i == vec.size() - 1) {
            std::cout << "(" << vec[i].first << ", " << vec[i].second << ")";
         } else {
            std::cout << "(" << vec[i].first << ", " << vec[i].second << ")"
                      << ",";
            // std::cout << vec[i] << ",";
         }
      }
      std::cout << "}" << std::endl;
   }

   void PrintChunk(std::vector<std::vector<std::pair<Long_t, Long_t>>> vec, int chunkNum)
   {
      std::vector<std::pair<Long_t, Long_t>> chunk = vec[chunkNum];

      PrintPair(chunk);
   }

   void SplitDataset()
   {
      // std::random_device rd;
      // std::mt19937 g(rd());

      std::mt19937 g(42);

      std::vector<Long_t> BlockSizes = {};

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

      std::vector<std::pair<Long_t, Long_t>> UnpermutedBlockIntervals(BlockIntervals.size());
      for (int i = 0; i < BlockIntervals.size(); ++i) {
         UnpermutedBlockIntervals[indices[i]] = BlockIntervals[i];
      }

      fTraining->BlockIntervals.insert(fTraining->BlockIntervals.begin(), UnpermutedBlockIntervals.begin(),
                                       UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks);
      fValidation->BlockIntervals.insert(fValidation->BlockIntervals.begin(),
                                         UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks,
                                         UnpermutedBlockIntervals.end());

      // PrintVector(indices) ;
      // PrintVector(BlockSizes);
      // PrintVector(PermutedBlockSizes);
      // PrintVector(BlockBoundaries);
      // PrintPair(BlockIntervals);
      PrintPair(UnpermutedBlockIntervals);

      fTraining->DistributeBlockIntervals();
      fValidation->DistributeBlockIntervals();

      std::cout << "========================= Before shiuffling =============================" << std::endl;
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

   void CreateTrainingChunksIntervals()
   {

      std::mt19937 g(42);

      std::shuffle(fTraining->FullBlockIntervalsInFullChunks.begin(), fTraining->FullBlockIntervalsInFullChunks.end(),
                   g);
      std::shuffle(fTraining->LeftoverBlockIntervalsInFullChunks.begin(),
                   fTraining->LeftoverBlockIntervalsInFullChunks.end(), g);
      std::shuffle(fTraining->FullBlockIntervalsInLeftoverChunks.begin(),
                   fTraining->FullBlockIntervalsInLeftoverChunks.end(), g);
      std::shuffle(fTraining->LeftoverBlockIntervalsInLeftoverChunks.begin(),
                   fTraining->LeftoverBlockIntervalsInLeftoverChunks.end(), g);

      fTraining->ChunksIntervals = {};
      fTraining->CreateChunksIntervals();

      std::shuffle(fTraining->ChunksIntervals.begin(), fTraining->ChunksIntervals.end(), g);

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

   void CreateValidationChunksIntervals()
   {

      std::mt19937 g(42);

      std::shuffle(fValidation->FullBlockIntervalsInFullChunks.begin(),
                   fValidation->FullBlockIntervalsInFullChunks.end(), g);
      std::shuffle(fValidation->LeftoverBlockIntervalsInFullChunks.begin(),
                   fValidation->LeftoverBlockIntervalsInFullChunks.end(), g);
      std::shuffle(fValidation->FullBlockIntervalsInLeftoverChunks.begin(),
                   fValidation->FullBlockIntervalsInLeftoverChunks.end(), g);
      std::shuffle(fValidation->LeftoverBlockIntervalsInLeftoverChunks.begin(),
                   fValidation->LeftoverBlockIntervalsInLeftoverChunks.end(), g);

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

   void PrintChunks()
   {
      std::cout << "Training: ";
      PrintPair(fTraining->BlockIntervals);
      std::cout << "Validation: ";
      PrintPair(fValidation->BlockIntervals);

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

   void LoadTrainingChunk(TMVA::Experimental::RTensor<float> &TrainChunkTensor, std::size_t chunk)
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

            if (fNotFiltered) {            
               RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
               ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);
               f_rdf.Foreach(func, fCols);
               chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
            }
            
            else {            
               std::size_t blockSize = BlocksInChunk[i].second - BlocksInChunk[i].first;
               std::cout << "Block size: " << blockSize << std::endl;
               std::cout << "Block: (" << BlocksInChunk[i].first << ", " << BlocksInChunk[i].second << ")" << std::endl;               
               for (std::size_t j = 0; j < blockSize; j++) {
                  RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
                  ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, (*fEntries)[BlocksInChunk[i].first + j], (*fEntries)[BlocksInChunk[i].first + j + 1]);
                  std::cout << (*fEntries)[BlocksInChunk[i].first + j] << " " << (*fEntries)[BlocksInChunk[i].first + j + 1] << std::endl;
                  f_rdf.Foreach(func, fCols);
                  chunkEntry++;
               }
            }
         }

         // shuffle data in RTensor
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      TrainChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   void LoadValidationChunk(TMVA::Experimental::RTensor<float> &ValidationChunkTensor, std::size_t chunk)
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

            if (fNotFiltered) {            
               RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
               ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);
               f_rdf.Foreach(func, fCols);
               chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
            }

            else {            
               std::size_t blockSize = BlocksInChunk[i].second - BlocksInChunk[i].first;
               std::cout << "Block size: " << blockSize << std::endl;
               std::cout << "Block: (" << BlocksInChunk[i].first << ", " << BlocksInChunk[i].second << ")" << std::endl;               
               for (std::size_t j = 0; j < blockSize; j++) {
                  RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
                  ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, (*fEntries)[BlocksInChunk[i].first + j], (*fEntries)[BlocksInChunk[i].first + j + 1]);
                  std::cout << (*fEntries)[BlocksInChunk[i].first + j] << " " << (*fEntries)[BlocksInChunk[i].first + j + 1] << std::endl;
                  f_rdf.Foreach(func, fCols);
                  chunkEntry++;
               }
            }
         }

         // shuffle data in RTensor
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                      ValidationChunkTensor.GetData() + i * fNumCols);
         }
      }
   }

   std::vector<std::size_t> GetTrainingChunkSizes() { return fTraining->ChunksSizes; }
   std::vector<std::size_t> GetValidationChunkSizes() { return fValidation->ChunksSizes; }

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
