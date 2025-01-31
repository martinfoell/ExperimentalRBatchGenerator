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

  std::vector<Long_t> fPartialSumRangeSizes;
  
  std::vector<std::pair<Long64_t,Long64_t>> fFullRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderTrainRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fReminderValidationRanges;

  std::vector<std::pair<Long64_t,Long64_t>> fTrainRanges;
  std::vector<std::pair<Long64_t,Long64_t>> fValidationRanges;
  
  ROOT::RDataFrame &f_rdf;
  std::vector<std::string> fCols;
  std::size_t fNumCols;

  bool fNotFiltered;
  bool fShuffle;

 public:
  RSplitTrainValidation(ROOT::RDataFrame &rdf, const std::size_t chunkSize, const std::size_t rangeSize,
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

    std::vector<Long_t> RangeSizes = {};
    RangeSizes.insert(RangeSizes.end(), fTotNumFullRanges, fRangeSize);
    RangeSizes.insert(RangeSizes.end(), fTotNumReminderRanges , fFullChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderTrainChunkReminderRanges , fReminderTrainChunkReminderRangeSize);
    RangeSizes.insert(RangeSizes.end(), fNumReminderValidationChunkReminderRanges, fReminderValidationChunkReminderRangeSize);        

    // std::cout << "{";
    // for (auto i : RangeSizes) {
    //   std::cout << i << ", ";
    // }
    // std::cout << "}" << std::endl;
    
    std::shuffle(RangeSizes.begin(), RangeSizes.end(), g);


    fPartialSumRangeSizes.resize(RangeSizes.size());

    std::partial_sum(RangeSizes.begin(), RangeSizes.end(), fPartialSumRangeSizes.begin());
    fPartialSumRangeSizes.insert(fPartialSumRangeSizes.begin(), 0);
  };

  void SplitRangeVector() {

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
        fReminderTrainRanges.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));      
      }
      
      else if (fPartialSumRangeSizes[i+1] - fPartialSumRangeSizes[i] == fReminderValidationChunkReminderRangeSize) {
        fReminderValidationRanges.push_back(std::make_pair(fPartialSumRangeSizes[i], fPartialSumRangeSizes[i+1]));      
      }
    }

    std::shuffle(fFullRanges.begin(), fFullRanges.end(), g);
    std::shuffle(fReminderRanges.begin(), fReminderRanges.end(), g);

    // move pairs if reminder is equal
    // if ()
    if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
         fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
         fFullChunkReminderRangeSize != 0 ) {
      fReminderTrainRanges.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      fReminderValidationRanges.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "i) Reminder range, reminder train range and reminder validation range are equal " << std::endl;
    }

    else if ( fFullChunkReminderRangeSize == fReminderTrainChunkReminderRangeSize and
              fFullChunkReminderRangeSize != 0) {
      fReminderTrainRanges.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "ii) Reminder range and reminder train range are equal " << std::endl;      
    }    

    else if ( fFullChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
              fFullChunkReminderRangeSize != 0) {
      fReminderValidationRanges.push_back(fReminderRanges.back());
      fReminderRanges.pop_back();

      std::cout << "iii) Reminder range and reminder validation range are equal " << std::endl;            
    }    

    else if ( fReminderTrainChunkReminderRangeSize == fReminderValidationChunkReminderRangeSize and
              fReminderTrainChunkReminderRangeSize != 0) {
      fReminderValidationRanges.push_back(fReminderTrainRanges.back());
      fReminderTrainRanges.pop_back();

      std::cout << "iv) Reminder train range and reminder validation range are equal " << std::endl;                  
    }    
  
  };    


  void CreateTrainValidationRangeVectors() {
    int currentElementFullRanges = 0;
    int currentElementReminderRanges = 0;

    // fill full chunk
    
    if (fNumFullTrainChunks != 0) {
      for (int i = 0; i < fNumFullTrainChunks; i++ ) {
        // fill full ranges 
        std::move(fFullRanges.begin(), fFullRanges.begin() + fNumFullChunkFullRanges, std::back_inserter(fTrainRanges));
        fFullRanges.erase(fFullRanges.begin(), fFullRanges.begin() + fNumFullChunkFullRanges);

        // fill reminder ranges
        if (fNumFullChunkReminderRanges != 0) {
          std::move(fReminderRanges.begin(), fReminderRanges.begin() + 1, std::back_inserter(fTrainRanges));        
          fReminderRanges.erase(fReminderRanges.begin(), fReminderRanges.begin() + 1);        
        };
      }
    }

    // fill reminder chunk

    // fill full ranges 
    if (fNumReminderTrainChunkFullRanges != 0) {
      std::move(fFullRanges.begin(), fFullRanges.begin() + fNumReminderTrainChunkFullRanges, std::back_inserter(fTrainRanges));
      fFullRanges.erase(fFullRanges.begin(), fFullRanges.begin() + fNumReminderTrainChunkFullRanges);
    }
    // fill reminder ranges
    if (fNumReminderTrainChunkReminderRanges != 0) {
      std::move(fReminderTrainRanges.begin(), fReminderTrainRanges.end(), std::back_inserter(fTrainRanges));
      fReminderTrainRanges.erase(fReminderTrainRanges.begin(), fReminderTrainRanges.end());
    }


    // fill full chunk
    
    if (fNumFullValidationChunks != 0) {    
      for (int i = 0; i < fNumFullValidationChunks; i++ ) {
        // fill full ranges 
        std::move(fFullRanges.begin(), fFullRanges.begin() + fNumFullChunkFullRanges, std::back_inserter(fValidationRanges));
        fFullRanges.erase(fFullRanges.begin(), fFullRanges.begin() + fNumFullChunkFullRanges);

        // fill reminder ranges
        if (fNumFullChunkReminderRanges != 0) {
          std::move(fReminderRanges.begin(), fReminderRanges.begin() + 1, std::back_inserter(fValidationRanges));        
          fReminderRanges.erase(fReminderRanges.begin(), fReminderRanges.begin() + 1);        
        };
      }
    }

    // fill reminder chunk

    // fill full ranges 
    if (fNumReminderValidationChunkFullRanges != 0) {
      std::move(fFullRanges.begin(), fFullRanges.begin() + fNumReminderValidationChunkFullRanges, std::back_inserter(fValidationRanges));
      fFullRanges.erase(fFullRanges.begin(), fFullRanges.begin() + fNumReminderValidationChunkFullRanges);
    }
    // fill reminder ranges
    if (fNumReminderValidationChunkReminderRanges != 0) {
      std::move(fReminderValidationRanges.begin(), fReminderValidationRanges.end(), std::back_inserter(fValidationRanges));
      fReminderValidationRanges.erase(fReminderValidationRanges.begin(), fReminderValidationRanges.end());
    }
  }

  void Start() {
    CreateRangeVector();
    SplitRangeVector();
    CreateTrainValidationRangeVectors();
  }
  
  void LoadTrainingDataset(TMVA::Experimental::RTensor<float> &TrainTensor) {
    TMVA::Experimental::RTensor<float> Tensor({fNumTrainEntries, fNumCols});     
    TrainTensor = TrainTensor.Resize({{fNumTrainEntries, fNumCols}});

    std::vector<int> indices(fNumTrainEntries);
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 1, 2, ..., 100

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
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 1, 2, ..., 100

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

  void PrintTrainValidationVector() {
    CreateRangeVector();
    SplitRangeVector();

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
    
    std::cout << "Reminder Train ranges: " << fReminderTrainRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;      
    for(auto i: fReminderTrainRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              

    std::cout << "Reminder Validation ranges: " << fReminderValidationRanges.size() << std::endl;
    std::cout << "-------------------------" << std::endl;
    for(auto i: fReminderValidationRanges) {
      std::cout << "(" << i.first << ", " << i.second << ")" << ", ";    
    }
    std::cout << " " << std::endl;
    std::cout << "-------------------------" << std::endl;
    std::cout << " " << std::endl;              
    
    CreateTrainValidationRangeVectors();

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

    PrintRowHeader("Chunk distribution", "Number", "Size", "Number", "Size", colWidthS, colWidth);  
    PrintRow("Full", fNumFullTrainChunks, fChunkSize, fNumFullValidationChunks, fChunkSize, colWidthS, colWidth);
    PrintRow("Reminder", fNumReminderTrainChunks, fReminderTrainChunkSize, fNumReminderValidationChunks, fReminderValidationChunkSize, colWidthS, colWidth);
    std::cout << std::string(colWidthS + 4 * colWidth, '-') << std::endl;        
    std::cout << std::string(colWidthS + 4 * colWidth, ' ') << std::endl;    

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
