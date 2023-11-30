#include "io.h"
#include <cstring>

int main(int argc, char** argv){
    if (argc!=3){
        std::cout<<"usage: ./txt2spmat txt_file save_path"<<std::endl;
    }

    std::string labels_file_to_use = argv[1];
    std::string save_path = argv[2];
    std::string universal_label = "0";

    std::vector<label_set> point_ids_to_labels;
    std::unordered_map<std::string, uint32_t> labels_to_number_of_points;
    label_set all_labels;

    std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) = parse_label_file(labels_file_to_use, universal_label);
    int64_t rows = point_ids_to_labels.size(), cols = all_labels.size(), nnz = 0;
    for (label_set& label:point_ids_to_labels){
        nnz+=label.size();
    }

    int64_t* row_index = new int64_t[rows+1];
    int32_t* col_index = new int32_t[nnz];
    float* value = new float[nnz];
    memset(value,0,sizeof(float)*nnz);

    int64_t count = 0;
    for (size_t i=0;i<rows;i++){
        row_index[i] = count;
        uint32_t j = 0;
        for (std::string label:point_ids_to_labels[i]){
            col_index[count+j] = atoi(label.c_str());
            j++;
        }
        count+=j;
    }
    row_index[rows] = count;

    write_sparse_matrix(save_path.c_str(),rows,cols,nnz,row_index,col_index,value);

}