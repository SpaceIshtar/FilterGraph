# Tag-Filtered ANNS
This is the repository of my ICDE25 paper Tag-Filtered Approximate Nearest Neighbor Search

## Usage
To build the repo:
```
mkdir build
cd build
cmake ..
make -j
```

To build Local Method Index:
```
./build_index data_path label_file data_type graph_degree num_of_entry_points Lsize ef_construction save_path
```

To build Global Method Index:
```
./build_index_v3 data_path label_file data_type degree_budget num_of_entry_points ef_construction save_path
```

To build Packing Strategy Index:
```
./build_index data_path label_file data_type graph_degree num_of_entry_points Lsize ef_construction pack_k threshold_high save_path
```

To search Global Method Index:
```
./search_index_v3 query_path query_label_spmat index_prefix gt_file topk ef_search data_type
```

To search Local/Packing Strategy Index:
* For local method, include "index.h" and do not include "index_sharing.h" in search_index.cpp
* For packing strategy, include "index_sharing.h" and do not include "index.h"
```
./search_index query_path query_label_spmat index_prefix gt_file topk ef_search data_type
```

## Data Format
* Vector bin file (datapath, querypath): the first two uint32 should be the number of vectors (nd) and dimension (dim) respectively. Then the following sizeof(datatype)\*nd\*dim bytes will be the vectors.

* Groundtruth format: the first two uint32 should be the number of queries (nq) and K respectively. Then the following sizeof(uint32)\*nq\*K are the ID of groundtruth. Then following sizeof(float)\*nq\*K are the distance between the query and the groundtruth.

* Tags file format: Tags should be stored in CSR format. Specifically, the file should first contain three int64 number rows, cols and nnz. Then the following sizeof(int64_t)\*(rows+1) represents the row index, followed by sizeof(int32_t)\*nnz bytes representing and column index and sizeof(float)\*nnz representing the values.