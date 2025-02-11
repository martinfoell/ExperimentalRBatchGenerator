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
    fChunkTensor.GetData()[fOffset++ + numColumns*i] = val;
  }
  
 public:
  RRangeChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor, int i, int numColumns)
    : fChunkTensor(chunkTensor),
      fI(i),
      fNumColumns(numColumns)
  {
  }

  void operator()( const ColTypes &...cols)
  {
    fVecSizeIdx = 1;
    (AssignToTensorRange(cols, fI, fNumColumns), ...);
  }
  
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
  
  std::vector<std::pair<Long64_t,Long64_t>> fFullRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fTrainRangesReminder;
  std::vector<std::pair<Long64_t,Long64_t>> fValidationRangesReminder;

  std::vector<std::pair<Long64_t,Long64_t>> fTrainRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fValidationRanges;

  std::vector<std::pair<Long64_t,Long64_t>> fFullTrainRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderTrainRanges;  

  std::vector<std::pair<Long64_t,Long64_t>> fFullValidationRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderValidationRanges;  
    
  ROOT::RDF::RNode &f_rdf;
  // ROOT::RDataFrame &f_rdf;
  std::vector<std::string> fCols;
  std::size_t fNumCols;

  bool fNotFiltered;
  bool fShuffle;

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

    // number of full chunks for training and validetion
    fNumFullTrainChunks = fNumTrainEntries / fChunkSize;
    fNumFullValidationChunks = fNumValidationEntries / fChunkSize;

    // total number of chunks from the dataset
    fNumFullChunkRanges = fNumFullTrainChunks + fNumFullValidationChunks;
    
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
    fNumReminderValidationChunkFullRanges = fReminderValidationChunkSize / fRangeSize;;

    // number of reminder ranges in the reminder chunk for training and validation
    fNumReminderTrainChunkReminderRanges = fReminderTrainChunkReminderRangeSize == 0 ? 0 : 1;
    fNumReminderValidationChunkReminderRanges = fReminderValidationChunkReminderRangeSize == 0 ? 0 : 1;

    // total number of ranges in the reminder chunk for training and validation
    fNumReminderTrainChunkRanges = fNumReminderTrainChunkFullRanges + fNumReminderTrainChunkReminderRanges;
    fNumReminderValidationChunkRanges = fNumReminderValidationChunkFullRanges + fNumReminderValidationChunkReminderRanges;
    
    // total number of full and reminder chunks in the dataset (train + val)
    fTotNumFullChunks = fNumFullTrainChunks + fNumFullValidationChunks;
    fTotNumReminderChunks = fNumReminderTrainChunks + fNumReminderValidationChunks;
    
    fTotNumReminderChunkFullRanges = fNumReminderTrainChunkFullRanges + fNumReminderValidationChunkFullRanges;
      
    fTotNumTrainFullRanges = fNumFullTrainChunks * fNumFullChunkFullRanges + fNumReminderTrainChunkFullRanges;
    fTotNumValidationFullRanges = fNumFullValidationChunks * fNumFullChunkFullRanges + fNumReminderValidationChunkFullRanges;

    fTotNumTrainReminderRanges = fNumFullTrainChunks * fNumFullChunkReminderRanges;
    fTotNumValidationReminderRanges = fNumFullValidationChunks * fNumFullChunkReminderRanges;    
    
    fTotNumFullRanges = fTotNumTrainFullRanges + fTotNumValidationFullRanges;
    fTotNumReminderRanges = fTotNumTrainReminderRanges + fTotNumValidationReminderRanges;

    fTotEntriesFromRanges = fTotNumFullRanges * fRangeSize + fTotNumReminderRanges * fFullChunkReminderRangeSize + fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize +  fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize;
  }

  void CreateRangeVector() {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<Long_t> RangeSizes = {};
    RangeSizes.insert(RangeSizes.end(), fTotNumFullRanges, fRangeSize);
    RangeSizes.insert(RangeSizes.end(), fTotNumReminderRanges , fFullChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderTrainChunkReminderRanges , fReminderTrainChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize);        

    std::shuffle(RangeSizes.begin(), RangeSizes.end(), g);

    fPartialSumRangeSizes.resize(RangeSizes.size());

    std::partial_sum(RangeSizes.begin(), RangeSizes.end(), fPartialSumRangeSizes.begin());
    fPartialSumRangeSizes.insert(fPartialSumRangeSizes.begin(), 0);
  };

  
  void SortRangeVector() {

    std::random_device rd;
    std::mt19937 g(rd());

    for (int i = 0; i < fPartialSumRangeSizes.size() - 1; i++) {
      if (fPartialSumRangeSizes[i+1] - fPartialSumRangeSizes[i] == fRangeSize) {
        fFullRanges.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));
      }
      
      else if (fPartialSumRangeSizes[i+1] - fPartialSumRangeSizes[i] == fFullChunkReminderRangeSize) {
        fReminderRanges.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));      
      }
      
      else if (fPartialSumRangeSizes[i+1] - fPartialSumRangeSizes[i] == fReminderTrainChunkReminderRangeSize) {
        fTrainRangesReminder.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));      
      }
      
      else if (fPartialSumRangeSizes[i+1] - fPartialSumRangeSizes[i] == fReminderValidationChunkReminderRangeSize) {
        fValidationRangesReminder.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));      
      }
    }

    std::shuffle(fFullRanges.begin(), fFullRanges.end(), g);
    std::shuffle(fReminderRanges.begin(), fReminderRanges.end(), g);

    // corner cases if some of the reminder sizes are equal
    if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
         fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
         fFullChunkReminderRangeSize != 0 ) {
      fTrainRangesReminder.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      fValidationRangesReminder.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "i) Reminder range, reminder train range and reminder validation range are equal " << std::endl;
    }

    else if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
              fFullChunkReminderRangeSize != 0) {
      fTrainRangesReminder.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "ii) Reminder range and reminder train range are equal " << std::endl;      
    }    

    else if ( fFullChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
              fFullChunkReminderRangeSize != 0) {
      fValidationRangesReminder.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "iii) Reminder range and reminder validation range are equal " << std::endl;            
    }    

    else if ( fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
              fReminderTrainChunkReminderRangeSize != 0) {
      fValidationRangesReminder.push_back(fTrainRangesReminder.back());
      fTrainRangesReminder.pop_back();

      std::cout << "iv) Reminder train range and reminder validation range are equal " << std::endl;                  
    }    

    
    // change to span later
    fFullTrainRanges = std::vector<std::pair<Long64_t,Long64_t>>(fFullRanges.begin(), fFullRanges.begin() + fTotNumTrainFullRanges);
    fReminderTrainRanges = std::vector<std::pair<Long64_t,Long64_t>>(fReminderRanges.begin(), fReminderRanges.begin() + fTotNumTrainReminderRanges);    

    fFullValidationRanges = std::vector<std::pair<Long64_t,Long64_t>>(fFullRanges.begin() + fTotNumTrainFullRanges, fFullRanges.begin() + fTotNumTrainFullRanges + fTotNumValidationFullRanges);
    fReminderValidationRanges = std::vector<std::pair<Long64_t,Long64_t>>(fReminderRanges.begin() + fTotNumTrainReminderRanges, fReminderRanges.begin() + fTotNumTrainReminderRanges + fTotNumValidationReminderRanges);    
    
  };    


  void CreateTrainRangeVector() {

    std::random_device rd;
    std::mt19937 g(rd());
    
    fTrainRanges = {};

    std::shuffle(fFullTrainRanges.begin(), fFullTrainRanges.end(), g);
    std::shuffle(fReminderTrainRanges.begin(), fReminderTrainRanges.end(), g);

    // Fill full chunks
    for (std::size_t i = 0; i < fNumFullTrainChunks; i++) {
      std::size_t startFull = i*fNumFullChunkFullRanges;
      std::size_t endFull = (i + 1)*fNumFullChunkFullRanges;

      std::size_t startReminder = i*fNumFullChunkReminderRanges;
      std::size_t endReminder = (i + 1)*fNumFullChunkReminderRanges;
      
      std::move(fFullTrainRanges.begin() + startFull, fFullTrainRanges.begin() + endFull, std::back_inserter(fTrainRanges));
      std::move(fReminderTrainRanges.begin() + startReminder, fReminderTrainRanges.begin() + endReminder, std::back_inserter(fTrainRanges));      
    }

    // Fill reminder chunks
    std::size_t startFullReminder = fNumFullTrainChunks*fNumFullChunkFullRanges;
    std::size_t endFullReminder = startFullReminder + fNumReminderTrainChunkFullRanges;    

    std::size_t endReminderReminder = fNumReminderTrainChunkReminderRanges;
    std::move(fFullTrainRanges.begin() + startFullReminder, fFullTrainRanges.begin() + endFullReminder, std::back_inserter(fTrainRanges));      
    std::move(fTrainRangesReminder.begin(), fTrainRangesReminder.begin() + endReminderReminder, std::back_inserter(fTrainRanges));
    
  }

  void CreateValidationRangeVector() {

    std::random_device rd;
    std::mt19937 g(rd());
    
    fValidationRanges = {};

    std::shuffle(fFullValidationRanges.begin(), fFullValidationRanges.end(), g);
    std::shuffle(fReminderValidationRanges.begin(), fReminderValidationRanges.end(), g);

    // Fill full chunks
    for (std::size_t i = 0; i < fNumFullValidationChunks; i++) {
      std::size_t startFull = i*fNumFullChunkFullRanges;
      std::size_t endFull = (i + 1)*fNumFullChunkFullRanges;

      std::size_t startReminder = i*fNumFullChunkReminderRanges;
      std::size_t endReminder = (i + 1)*fNumFullChunkReminderRanges;
      std::move(fFullValidationRanges.begin() + startFull, fFullValidationRanges.begin() + endFull, std::back_inserter(fValidationRanges));
      std::move(fReminderValidationRanges.begin() + startReminder, fReminderValidationRanges.begin() + endReminder, std::back_inserter(fValidationRanges));      
    }

    // Fill reminder chunk
    std::size_t startFullReminder = fNumFullValidationChunks*fNumFullChunkFullRanges;
    std::size_t endFullReminder = startFullReminder + fNumReminderValidationChunkFullRanges;    

    std::size_t endReminderReminder = fNumReminderValidationChunkReminderRanges;
    std::move(fFullValidationRanges.begin() + startFullReminder , fFullValidationRanges.begin() + endFullReminder, std::back_inserter(fValidationRanges));          
    std::move(fValidationRangesReminder.begin(), fValidationRangesReminder.begin() + endReminderReminder, std::back_inserter(fValidationRanges));

  }
  

  void Start() {
    CreateRangeVector();
    SortRangeVector();
  }
  
  void LoadTrainingDataset(TMVA::Experimental::RTensor<float> &TrainTensor) {
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

  void LoadValidationDataset(TMVA::Experimental::RTensor<float> &ValidationTensor) {
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


  void LoadTrainChunk(TMVA::Experimental::RTensor<float> &TrainChunkTensor, std::size_t chunk) {

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
        std::size_t entry = chunk*fNumFullChunkRanges + i;
      
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
        std::size_t entry = chunk*fNumFullChunkRanges + i;
      
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

  void LoadValidationChunk(TMVA::Experimental::RTensor<float> &ValidationChunkTensor, std::size_t chunk) {

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
        std::size_t entry = chunk*fNumFullChunkRanges + i;
      
        RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
        ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fValidationRanges[entry].first, fValidationRanges[entry].second);
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
        std::size_t entry = chunk*fNumFullChunkRanges + i;
      
        RRangeChunkLoaderFunctor<Args...> func(Tensor, chunkEntry, fNumCols);
        ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, fValidationRanges[entry].first, fValidationRanges[entry].second);
        f_rdf.Foreach(func, fCols);
        chunkEntry += fValidationRanges[entry].second - fValidationRanges[entry].first;
      }

      for (std::size_t i = 0; i < fReminderValidationChunkSize; i++) {
        std::copy(Tensor.GetData() + indices[i] * fNumCols, Tensor.GetData() + (indices[i] + 1) * fNumCols,
                  ValidationChunkTensor.GetData() + i * fNumCols);
      
      }
    }
    
  }

  void CheckIfUnique(TMVA::Experimental::RTensor<float> &Tensor) {
    auto tensorSize = Tensor.GetSize();
    TMVA::Experimental::RTensor<float> SqueezeTensor  = Tensor.Reshape({1, tensorSize}).Squeeze();

    std::list<int> allEntries;
    for (int i = 0; i < tensorSize; i++) {
      allEntries.push_back(SqueezeTensor(0,i));
    }
    allEntries.sort();
    allEntries.unique();
    if (allEntries.size() == tensorSize) {
      std::cout << "Tensor consists of only unique elements" << std::endl;
    }
  };

  void CheckIfOverlap(TMVA::Experimental::RTensor<float> &Tensor1, TMVA::Experimental::RTensor<float> &Tensor2) {  
    auto tensorSize1 = Tensor1.GetSize();
    TMVA::Experimental::RTensor<float> SqueezeTensor1  = Tensor1.Reshape({1, tensorSize1}).Squeeze();

    std::list<int> allEntries1;
    for (int i = 0; i < tensorSize1; i++) {
      allEntries1.push_back(SqueezeTensor1(0,i));
    }

    auto tensorSize2 = Tensor2.GetSize();
    TMVA::Experimental::RTensor<float> SqueezeTensor2  = Tensor2.Reshape({1, tensorSize2}).Squeeze();

    std::list<int> allEntries2;
    for (int i = 0; i < tensorSize2; i++) {
      allEntries2.push_back(SqueezeTensor2(0,i));
    }

    std::set<int> result;

    // Call the set_intersection(), which computes the
    // intersection of set1 and set2 and
    // inserts the result into the 'result' set
    std::set<int> set1(allEntries1.begin(), allEntries1.end());
    std::set<int> set2(allEntries2.begin(), allEntries2.end());        
    std::set_intersection(set1.begin(), set1.end(), set2.begin(),
                     set2.end(),
                     inserter(result, result.begin()));    
    // std::list<int> result = intersection(allEntries1, allEntries2);

    if (result.size() == 0) {
      std::cout << "No overlap between the tensors" << std::endl;
    }
    else {
      std::cout << "Intersection between tensors: ";
      for (int num : result) {
        std::cout << num << " ";
      }
      std::cout << std::endl;
    }
  };
  
  std::size_t GetNumTrainChunks() {
    return fNumTrainChunks;
  }

  std::size_t GetNumValidationChunks() {
    return fNumValidationChunks;
  }

  std::size_t GetNumberOfFullTrainingChunks() {
    return fNumFullTrainChunks;
  }
  
  
  void PrintTrainValidationVector() {

    std::cout << " " << std::endl;    
    std::cout << "Full ranges: " << fFullRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;
    for(auto i: fFullRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              

    std::cout << "Reminder ranges: " << fReminderRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;    
    for(auto i: fReminderRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              
    
    std::cout << "Reminder Train ranges: " << fTrainRangesReminder.size() << std::endl;
    std::cout << "-------------------------" << std::endl;      
    for(auto i: fTrainRangesReminder) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              

    std::cout << "Reminder Validation ranges: " << fValidationRangesReminder.size() << std::endl;
    std::cout << "-------------------------" << std::endl;
    for(auto i: fValidationRangesReminder) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              
    
    // CreateTrainValidationRangeVectors();

    std::cout << " " << std::endl;
    std::cout << "Train ranges: " << fTrainRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;    
    for(auto i: fTrainRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              

    std::cout << "Validation ranges: " << fValidationRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;
    for(auto i: fValidationRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              
    
    
  }

  
  void PrintRangeVector() {
    std::cout << "{";
    for (auto i : fRangeVector) {
      std::cout << i << ", ";
    }
    std::cout << "}" << std::endl;
  }
  
  //to do: make variables which count how many ranges of each sort
  
  void PrintRow(std::string title, int col1, int col2, int col3, int col4, int colWidthS, int colWidth) {
    std::cout << std::left;
    std::cout << std::setw(colWidthS) << title 
              << std::setw(colWidth) << col1
              << std::setw(colWidth) << col2
              << std::setw(colWidth) << col3
              << std::setw(colWidth) << col4
              << std::endl;
  };

  void PrintRowB(std::string title, int col1, int col2, int col3, int col4, int col5, int colWidthS, int colWidth) {
    std::cout << std::left;
    std::cout << std::setw(colWidthS) << title 
              << std::setw(colWidth) << col1
              << std::setw(colWidth) << col2
              << std::setw(colWidth) << col3
              << std::setw(colWidth) << col4
              << std::setw(colWidth) << col5      
              << std::endl;
  };
  
  void PrintRowHeader(std::string title, string col1, string col2, string col3, string col4, int colWidthS, int colWidth) {
    std::cout << std::string(colWidthS + 4 * colWidth, '=') << std::endl;
    std::cout << std::left;
    std::cout << std::setw(colWidthS) << title 
              << std::setw(colWidth) << col1
              << std::setw(colWidth) << col2
              << std::setw(colWidth) << col3
              << std::setw(colWidth) << col4
              << std::endl;
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;
    
  };

  void PrintRowHeaderB(std::string title, string col1, string col2, string col3, string col4, string col5, int colWidthS, int colWidth) {
    std::cout << std::string(colWidthS + 5 * colWidth, '=') << std::endl;
    std::cout << std::left;
    std::cout << std::setw(colWidthS) << title 
              << std::setw(colWidth) << col1
              << std::setw(colWidth) << col2
              << std::setw(colWidth) << col3
              << std::setw(colWidth) << col4
              << std::setw(colWidth) << col5      
              << std::endl;
    std::cout << std::string(colWidthS + 5 * colWidth, '-') << std::endl;
    
  };
  
  void PrintChunkDistributions() {

    const int colWidthS = 25;
    const int colWidth = 10;

    // std::cout << std::left;
    std::cout 
      << std::setw(colWidthS + colWidth) << "Train" 
      << std::setw(2*colWidth) << "Validation"
      << std::endl;

    PrintRowHeader("Entries", "Number", "Size", "Number", "Size", colWidthS, colWidth);  
    PrintRow("Total", 1, fNumTrainEntries, 1, fNumValidationEntries, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    PrintRowHeader("Chunk distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);  
    PrintRow("Full", fNumFullTrainChunks, fChunkSize, fNumFullValidationChunks, fChunkSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumReminderTrainChunks, fReminderTrainChunkSize, fNumReminderValidationChunks, fReminderValidationChunkSize, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

  }

  void PrintRangeDistributions() {

    const int colWidthS = 25;
    const int colWidth = 10;

    std::cout << std::left;
    std::cout << "Full chunks" << std::endl;
    PrintRowHeader("Range distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);          
    PrintRow("Full", fNumFullChunkFullRanges, fRangeSize, fNumFullChunkFullRanges, fRangeSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    std::cout << std::left;
    std::cout << "Reminder chunks" << std::endl;
    PrintRowHeader("Range distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);          
    PrintRow("Full", fNumReminderTrainChunkFullRanges, fRangeSize, fNumReminderValidationChunkFullRanges, fRangeSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumReminderTrainChunkReminderRanges, fReminderTrainChunkReminderRangeSize, fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize, colWidthS, colWidth);    
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    PrintRowHeaderB("Ranges", "Number", "Size", "Number", "Size", "Entries", colWidthS, colWidth);
    PrintRowB("Full",  fTotNumTrainFullRanges, fRangeSize, fTotNumValidationFullRanges, fRangeSize,
              fTotNumTrainFullRanges * fRangeSize +  fTotNumValidationFullRanges *fRangeSize, colWidthS, colWidth);
    PrintRowB("Reminder", fTotNumTrainReminderRanges, fFullChunkReminderRangeSize, fTotNumValidationReminderRanges, fFullChunkReminderRangeSize,
              fTotNumTrainReminderRanges * fFullChunkReminderRangeSize + fTotNumValidationReminderRanges * fFullChunkReminderRangeSize, colWidthS, colWidth);
    PrintRowB("Reminder Train", fNumReminderTrainChunkReminderRanges, fReminderTrainChunkReminderRangeSize, 0, 0,
              fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize, colWidthS, colWidth);
    PrintRowB("Reminder Validation", 0, 0, fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize,
              fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize, colWidthS, colWidth);

    std::cout << std::string(colWidthS + 5 * colWidth, '-') << std::endl;            
    PrintRowB("Total", fTotNumFullRanges * fRangeSize, fTotNumReminderRanges * fFullChunkReminderRangeSize, fNumReminderTrainChunkReminderRanges * fReminderTrainChunkReminderRangeSize, fNumReminderValidationChunkReminderRanges * fReminderValidationChunkReminderRangeSize, fTotEntriesFromRanges, colWidthS, colWidth);        
    
    std::cout << fTotNumFullRanges << " " << fTotNumReminderRanges << std::endl;

  }

  
};
