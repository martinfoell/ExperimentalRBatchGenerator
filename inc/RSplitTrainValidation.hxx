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

  std::vector<Long_t> fRangeVector;

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

  void CreateRangeVector() {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<std::pair<Long64_t,Long64_t>> FullRanges;
    std::vector<std::pair<Long64_t,Long64_t>> ReminderRanges;
    std::vector<std::pair<Long64_t,Long64_t>> ReminderTrainRanges;
    std::vector<std::pair<Long64_t,Long64_t>> ReminderValidationRanges;

    std::vector<std::pair<Long64_t,Long64_t>> TrainRanges;
    std::vector<std::pair<Long64_t,Long64_t>> ValidationRanges;
    std::vector<Long_t> RangeSizes = {};
    RangeSizes.insert(RangeSizes.end(), fTotNumFullRanges, fRangeSize);
    RangeSizes.insert(RangeSizes.end(), fTotNumReminderRanges , fFullChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderTrainChunkReminderRanges , fReminderTrainChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize);        

    std::cout << "{";
    for (auto i : RangeSizes) {
      std::cout << i << ", ";
    }
    std::cout << "}" << std::endl;
    
    std::shuffle(RangeSizes.begin(), RangeSizes.end(), g);

    std::vector<Long_t> PartialSumRangeSizes(RangeSizes.size());
    std::partial_sum(RangeSizes.begin(), RangeSizes.end(), PartialSumRangeSizes.begin());
    PartialSumRangeSizes.insert(PartialSumRangeSizes.begin(), 0);
    
    for (int i = 0; i < PartialSumRangeSizes.size() - 1; i++) {
      if (PartialSumRangeSizes[i+1] - PartialSumRangeSizes[i] == fRangeSize) {
        FullRanges.push_back(std::make_pair(PartialSumRangeSizes[i], PartialSumRangeSizes[i+1]));
      }
      
      else if (PartialSumRangeSizes[i+1] - PartialSumRangeSizes[i] == fFullChunkReminderRangeSize) {
        ReminderRanges.push_back(std::make_pair(PartialSumRangeSizes[i], PartialSumRangeSizes[i+1]));      
      }
      
      else if (PartialSumRangeSizes[i+1] - PartialSumRangeSizes[i] == fReminderTrainChunkReminderRangeSize) {
        ReminderTrainRanges.push_back(std::make_pair(PartialSumRangeSizes[i], PartialSumRangeSizes[i+1]));      
      }
      
      else if (PartialSumRangeSizes[i+1] - PartialSumRangeSizes[i] == fReminderValidationChunkReminderRangeSize) {
        ReminderValidationRanges.push_back(std::make_pair(PartialSumRangeSizes[i], PartialSumRangeSizes[i+1]));      
      }
    }

    std::shuffle(FullRanges.begin(), FullRanges.end(), g);
    std::shuffle(ReminderRanges.begin(), ReminderRanges.end(), g);

    // move pairs if reminder is equal
    if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
         fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize ) {
      ReminderTrainRanges.push_back(ReminderRanges.back());
      ReminderRanges.pop_back();

      ReminderValidationRanges.push_back(ReminderRanges.back());
      ReminderRanges.pop_back();

      std::cout << "i) Reminder range, reminder train range and reminder validation range are equal " << std::endl;
    }

    else if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize ) {
      ReminderTrainRanges.push_back(ReminderRanges.back());
      ReminderRanges.pop_back();

      std::cout << "ii) Reminder range and reminder train range are equal " << std::endl;      
    }    

    else if ( fFullChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize ) {
      ReminderValidationRanges.push_back(ReminderRanges.back());
      ReminderRanges.pop_back();

      std::cout << "iii) Reminder range and reminder validation range are equal " << std::endl;            
    }    

    else if ( fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize ) {
      ReminderValidationRanges.push_back(ReminderTrainRanges.back());
      ReminderTrainRanges.pop_back();

      std::cout << "iv) Reminder train range and reminder validation range are equal " << std::endl;                  
    }    
    

    int currentElementFullRanges = 0;
    int currentElementReminderRanges = 0;
    
    // Full train chunks
    for (int i = 0; i < fNumFullTrainChunks; i++ ) {
      TrainRanges.insert(TrainRanges.end(), FullRanges.begin() + fNumFullChunkFullRanges*i, FullRanges.begin() + fNumFullChunkFullRanges*(i + 1));
      if (fNumFullChunkReminderRanges != 0) {
        TrainRanges.insert(TrainRanges.end(), ReminderRanges.begin() + fNumFullChunkReminderRanges*i, ReminderRanges.begin() + fNumFullChunkReminderRanges*i + 1);        
      }
    }
    
    currentElementFullRanges = fNumFullTrainChunks*fNumFullChunkFullRanges;
    currentElementReminderRanges = fNumReminderTrainChunks*fNumFullChunkReminderRanges;
    
    // Reminder train chunk 
    TrainRanges.insert(TrainRanges.end(), FullRanges.begin() + currentElementFullRanges, FullRanges.begin() + currentElementFullRanges + fNumReminderTrainChunkFullRanges);
    TrainRanges.insert(TrainRanges.end(), ReminderTrainRanges.begin(), ReminderTrainRanges.end());
    
    currentElementFullRanges += fNumReminderTrainChunkFullRanges;

    // Full validation chunk
    for (int i = 0; i < fNumFullValidationChunks; i++ ) {
      ValidationRanges.insert(ValidationRanges.end(), FullRanges.begin() + currentElementFullRanges + fNumFullChunkFullRanges*i, FullRanges.begin() + currentElementFullRanges + fNumFullChunkFullRanges*(i + 1));
      if (fNumFullChunkReminderRanges != 0) {
        ValidationRanges.insert(ValidationRanges.end(), ReminderRanges.begin() + currentElementReminderRanges + fNumFullChunkReminderRanges*i, ReminderRanges.begin() + fNumFullChunkReminderRanges*i + 1);        
      }
    }
    
    currentElementFullRanges += fNumFullValidationChunks*fNumFullChunkFullRanges;
    currentElementReminderRanges = fNumFullTrainChunks*fNumFullChunkReminderRanges;
    
    // // Reminder train chunk 
    ValidationRanges.insert(ValidationRanges.end(), FullRanges.begin() + currentElementFullRanges, FullRanges.begin() + currentElementFullRanges + fNumReminderTrainChunkFullRanges);
    ValidationRanges.insert(ValidationRanges.end(), ReminderValidationRanges.begin(), ReminderValidationRanges.end());
    
    // currentElementFullRanges += fNumReminderTrainChunkFullRanges;
    
    std::cout << "{";
    for (auto i : PartialSumRangeSizes) {
      std::cout << i << ", ";
    }
    std::cout << "}" << std::endl;

    std::cout << "Train ranges: " << TrainRanges.size() << std::endl;
    for(auto i: TrainRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;

    std::cout << "Validation ranges: " << ValidationRanges.size() << std::endl;
    for(auto i: ValidationRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    
    std::cout << "Full ranges: " << FullRanges.size() << std::endl;
    for(auto i: FullRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;

    std::cout << "Reminder ranges: " << ReminderRanges.size() << std::endl;  
    for(auto i: ReminderRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;

    std::cout << "Reminder Train ranges: " << ReminderTrainRanges.size() << std::endl;  
    for(auto i: ReminderTrainRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;

    std::cout << "Reminder Validation ranges: " << ReminderValidationRanges.size() << std::endl;  
    for(auto i: ReminderValidationRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
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
