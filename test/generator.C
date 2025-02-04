// #include "../inc/RChunkLoader.hxx"
// #include "../inc/RBatchLoader.hxx"
#include "../inc/RBatchGenerator.hxx"



void generator() {


  ROOT::RDataFrame rdf("tree", "../data/file*.root");

  std::size_t chunkSize = 75;
  std::size_t rangeSize = 10;
  std::size_t batchSize = 10;  
  float validationSplit = 0.3;
  std::vector<std::string> columns = {"A"};
  bool shuffle = false;


  
  RBatchGenerator<Double_t> generator(rdf, chunkSize, rangeSize, batchSize, validationSplit, columns, shuffle);


  for (std::size_t i = 0; i < 10; i++) {
    auto batch = generator.GenerateTrainBatch();
    std::cout << "Batch " << i + 1 << ": " << std::endl;
    std::cout << batch << std::endl;
  }

  
}



