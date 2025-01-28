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


class RSplitTrainValidation {
 private:
  std::size_t fNumEntries;
  std::size_t fChunkSize;
  std::size_t fRangeSize;
  float fValidationSplit;

  std::size_t fNumTrainEntries;
  std::size_t fNumValidationEntries;  

  // chunks
  std::size_t fNumFullTrainChunks;
  std::size_t fNumFullValidationChunks;

  std::size_t fNumReminderTrainChunks;
  std::size_t fNumReminderValidationChunks;
  
  std::size_t fReminderTrainChunkSize;
  std::size_t fReminderValidationChunkSize;

  // ranges

  std::size_t fNumFullChunkFullRanges;
  std::size_t fNumFullChunkReminderRanges;  

  std::size_t fNumReminderTrainChunkFullRanges;
  std::size_t fNumReminderValidationChunkFullRanges;

  std::size_t fNumReminderTrainChunkReminderRanges;
  std::size_t fNumReminderValidationChunkReminderRanges;
  
  std::size_t fFullChunkReminderRangeSize;
  std::size_t fReminderTrainChunkReminderRangeSize;
  std::size_t fReminderValidationChunkReminderRangeSize;
  
  
  ROOT::RDataFrame &f_rdf;    

  bool fNotFiltered;

 public:
  RSplitTrainValidation(ROOT::RDataFrame &rdf, const std::size_t chunkSize, const std::size_t rangeSize,
                        const float validationSplit = 0.0)
    : f_rdf(rdf),
      fChunkSize(chunkSize),
      fRangeSize(rangeSize),
      fValidationSplit(validationSplit),
      fNotFiltered(f_rdf.GetFilterNames().empty())
  {
    if (fNotFiltered) {
      fNumEntries = f_rdf.Count().GetValue();
    }
    
    fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
    fNumTrainEntries = fNumEntries - fNumValidationEntries;

    // chunks
    fNumFullTrainChunks = fNumTrainEntries / fChunkSize;
    fNumFullValidationChunks = fNumValidationEntries / fChunkSize;

    fNumReminderTrainChunks = fReminderTrainChunkSize == 0 ? 0 : 1;
    fNumReminderValidationChunks = fReminderValidationChunkSize == 0 ? 0 : 1;
    
    fReminderTrainChunkSize = fNumTrainEntries % fChunkSize;
    fReminderValidationChunkSize = fNumValidationEntries % fChunkSize;

    // ranges
    fNumFullChunkFullRanges = fChunkSize / fRangeSize;
    fNumFullChunkReminderRanges = fFullChunkReminderRangeSize == 0 ? 0 : 1;

    fNumReminderTrainChunkFullRanges = fReminderTrainChunkSize / fRangeSize;
    fNumReminderValidationChunkFullRanges = fReminderValidationChunkSize / fRangeSize;;

    fNumReminderTrainChunkReminderRanges = fReminderTrainChunkReminderRangeSize == 0 ? 0 : 1;
    fNumReminderValidationChunkReminderRanges = fReminderValidationChunkReminderRangeSize == 0 ? 0 : 1;

    fFullChunkReminderRangeSize = fChunkSize % fRangeSize;
    
    fReminderTrainChunkReminderRangeSize = fReminderTrainChunkSize % fRangeSize;
    fReminderValidationChunkReminderRangeSize = fReminderValidationChunkSize % fRangeSize;

    
    
    
  }

  void PrintProperties() {

    
    std::cout << fNumEntries << " " << fChunkSize << " " << fRangeSize << std::endl;
    std::cout << fNumTrainEntries << " " << fNumValidationEntries << std::endl;
    std::cout << "Chunks" << std::endl;
    std::cout << "Reminder size " << fReminderTrainChunkSize << " " << fReminderValidationChunkSize << std::endl;
    std::cout << "Num full " << fNumFullTrainChunks << " " << fNumFullValidationChunks << std::endl;
    std::cout << "Num reminder " << fNumReminderTrainChunks << " " << fNumReminderValidationChunks << std::endl;
    std::cout << "Ranges" << std::endl;
    std::cout << "Reminder size " << fReminderTrainChunkReminderRangeSize << " " << fReminderValidationChunkSize << std::endl;
    std::cout << "Num full " << fNumFullTrainChunks << " " << fNumFullValidationChunks << std::endl;
    std::cout << "Num reminder " << fNumReminderTrainChunks << " " << fNumReminderValidationChunks << std::endl;
    
  }
};
