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
    
    // tot
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
  
  void PrintProperties() {

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

    PrintRowHeader("Chunk", "Number", "Size", "Number", "Size", colWidthS, colWidth);  
    PrintRow("Full", fNumFullTrainChunks, fChunkSize, fNumFullValidationChunks, fChunkSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumReminderTrainChunks, fReminderTrainChunkSize, fNumReminderValidationChunks, fReminderValidationChunkSize, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    
      
    PrintRowHeader("Full chunks range", "Number", "Size", "Number", "Size", colWidthS, colWidth);      
    PrintRow("Full", fNumFullChunkFullRanges, fRangeSize, fNumFullChunkFullRanges, fRangeSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, fNumFullChunkReminderRanges, fFullChunkReminderRangeSize, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

    PrintRowHeader("Rem. chunks range", "Number", "Size", "Number", "Size", colWidthS, colWidth);          
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
