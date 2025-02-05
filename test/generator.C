#include "../inc/RBatchGenerator_python.hxx"

void generator() {

  // ROOT::RDataFrame rdf("tree", "../data/bigfile.root");

  // std::size_t chunkSize = 20000;
  // std::size_t rangeSize = 10;
  // std::size_t batchSize = 1000;  
  // float validationSplit = 0.3;
  // std::vector<std::string> columns = {"A", "a", "b", "c"};
  // bool shuffle = false;

  ROOT::RDataFrame rdf("tree", "../data/file*.root");
  auto rdf_node = ROOT::RDF::AsRNode(rdf);
  
  std::size_t chunkSize = 75;
  std::size_t rangeSize = 10;
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A"};
  bool shuffle = false;

  std::size_t batchSize = 10;
  std::size_t maxBatches = 50;  
  
  RBatchGenerator<Double_t> generator(rdf_node, chunkSize, rangeSize, batchSize, validationSplit, shuffle, columns);

  for (std::size_t i = 0; i < 3; i++) {
    auto batch = generator.GenerateTrainBatch();
    std::cout << "Batch " << i + 1 << ": " << std::endl;
    // std::cout << batch << std::endl;
  }

  
}



