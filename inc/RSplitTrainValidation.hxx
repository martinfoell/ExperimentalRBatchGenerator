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

    fReminderTrainChunkSize = fNumTrainEntries % fChunkSize;
    fReminderValidationChunkSize = fNumValidationEntries % fChunkSize;
    
    fNumReminderTrainChunks = fReminderTrainChunkSize == 0 ? 0 : 1;
    fNumReminderValidationChunks = fReminderValidationChunkSize == 0 ? 0 : 1;
    

    // ranges
    fNumFullChunkFullRanges = fChunkSize / fRangeSize;
    fFullChunkReminderRangeSize = fChunkSize % fRangeSize;
    fNumFullChunkReminderRanges = fFullChunkReminderRangeSize == 0 ? 0 : 1;
    
    fReminderTrainChunkReminderRangeSize = fReminderTrainChunkSize % fRangeSize;
    fReminderValidationChunkReminderRangeSize = fReminderValidationChunkSize % fRangeSize;

    fNumReminderTrainChunkFullRanges = fReminderTrainChunkSize / fRangeSize;
    fNumReminderValidationChunkFullRanges = fReminderValidationChunkSize / fRangeSize;;

    fNumReminderTrainChunkReminderRanges = fReminderTrainChunkReminderRangeSize == 0 ? 0 : 1;
    fNumReminderValidationChunkReminderRanges = fReminderValidationChunkReminderRangeSize == 0 ? 0 : 1;
    
    
  }

  void PrintRow(std::string title, int col1, int col2, int col3, int col4, int colWidthS, int colWidth) {
    std::cout << std::left;
    std::cout << std::setw(colWidthS) << title 
              << std::setw(colWidth) << col1
              << std::setw(colWidth) << col2
              << std::setw(colWidth) << col3
              << std::setw(colWidth) << col4
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
  
  void PrintProperties() {

    const int colWidthS = 25;
    const int colWidth = 15;

    // std::cout << std::left;
    std::cout 
      << std::setw(colWidthS + colWidth) << "Train" 
      << std::setw(2*colWidth) << "Validation"
      << std::endl;

    PrintRowHeader("Entries", "Size", "Number", "Size", "Number", colWidthS, colWidth);  
    PrintRow("Total", fNumTrainEntries, 1, fNumValidationEntries, 1, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    PrintRowHeader("Chunk", "Size", "Number", "Size", "Number", colWidthS, colWidth);
    PrintRow("Full", fChunkSize, fNumFullTrainChunks, fChunkSize, fNumFullValidationChunks, colWidthS, colWidth);
    PrintRow("Reminder", fReminderTrainChunkSize, fNumReminderTrainChunks, fReminderValidationChunkSize, fNumReminderValidationChunks, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    
      
    PrintRowHeader("Full chunks range", "Size", "Number", "Size", "Number", colWidthS, colWidth);
    PrintRow("Full", fRangeSize, fNumFullChunkFullRanges, fRangeSize, fNumFullChunkFullRanges, colWidthS, colWidth);
    PrintRow("Reminder", fFullChunkReminderRangeSize, fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, fNumFullChunkReminderRanges, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    PrintRowHeader("Rem. chunks range", "Size", "Number", "Size", "Number", colWidthS, colWidth);
    PrintRow("Full", fRangeSize, fNumReminderTrainChunkFullRanges, fRangeSize, fNumReminderValidationChunkFullRanges, colWidthS, colWidth);
    PrintRow("Reminder", fReminderTrainChunkReminderRangeSize, fNumReminderTrainChunkReminderRanges, fReminderValidationChunkReminderRangeSize, fNumReminderValidationChunkReminderRanges, colWidthS, colWidth);    
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    
    
  }
};
