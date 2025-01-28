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


class RSplitTrainValidation {
 private:
  std::size_t fNumEntries;
  float fValidationSplit;
  std::size_t fChunkSize;
  std::size_t fRangeSize;  

  ROOT::RDataFrame &f_rdf;    

  bool fNotFiltered;

 public:
  RSplitTrainValidation(ROOT::RDataFrame &rdf, const std::size_t chunkSize, const std::size_t batchSize,
                        const float validationSplit = 0.0, bool dropRemainder = true)
    : f_rdf(rdf),
      fChunkSize(chunkSize),
      fBatchSize(batchSize),
      fValidationSplit(validationSplit),
      fNotFiltered(f_rdf.GetFilterNames().empty()),
  {

  }
