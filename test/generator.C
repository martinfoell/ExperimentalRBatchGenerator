#include "../inc/RBatchGenerator_python.hxx"

void generator() {

  ROOT::RDataFrame rdf("tree", "../data/bigfile.root");

  std::size_t chunkSize = 2000000;
  std::size_t rangeSize = 10;
  std::size_t batchSize = 1000;  
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A", "a", "b", "c"};
  bool shuffle = false;
  
  RBatchGenerator<Double_t, Double_t, Double_t, Double_t> generator(rdf, chunkSize, rangeSize, batchSize, validationSplit, columns, shuffle);

  for (std::size_t i = 0; i < 300; i++) {
    auto batch = generator.GenerateTrainBatch();
    std::cout << "Batch " << i + 1 << ": " << std::endl;
    // std::cout << batch << std::endl;
  }

  
}



