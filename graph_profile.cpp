#include "index.h"

template<typename T>
void graph_profile(std::string& index_prefix, Distance<T>* dist_fn){
    FilterIndex<T>* index = new FilterIndex<T>(dist_fn);
    index->load(index_prefix);
    std::cout<<"Load Index Ends."<<std::endl;
    size_t nd = index->_nd;
    size_t dim = index->_dim;
    size_t graph_offset = index->_graph_offset;
    size_t graph_degree = index->_graph_degree;
    size_t point_size = index->_point_size;
    float avg_degree = 0;
    size_t min_degree = graph_degree;
    size_t max_degree = 0;
    for (size_t i=0;i<1010;i++){
        for (int j=0;j<index->_graph_row_index[i].size();j++){
            auto p = index->_graph_row_index[i][j];
            std::cout<<"("<<p.first<<", "<<p.second<<"); ";
        }
        std::cout<<std::endl;
    //     uint32_t* neighbor = (uint32_t*)(index->_storage+i*point_size+graph_offset);
    //     size_t cur_degree = 0;
    //     std::cout<<"neighbors: ";
    //     for (uint32_t j=0;j<graph_degree;j++){
            
    //         if (neighbor[j]<nd){
    //             cur_degree++;
    //             std::cout<<neighbor[j]<<" ";
    //         }
    //     }
    //     std::cout<<std::endl;
    //     avg_degree+=cur_degree;
    //     if (cur_degree<min_degree) min_degree = cur_degree;
    //     if (cur_degree>max_degree) max_degree = cur_degree;
    //     std::cout<<"point "<<i<<", degree: "<<cur_degree<<std::endl;
    }
    // avg_degree = avg_degree/(float)nd;
    // std::cout<<"max degree: "<<max_degree<<", min degree: "<<min_degree<<", avg_degree: "<<avg_degree<<std::endl;
}

int main(int argc, char** argv){
    if (argc!=3){
        std::cout<<"usage: ./graph_profile incex_prefix data_type"<<std::endl;
        exit(-1);
    }
    std::string index_prefix = argv[1];
    std::string data_type = argv[2];

    if (data_type=="float"){
        Distance<float>* dist_fn = new AVXDistanceL2Float();
        graph_profile<float>(index_prefix,dist_fn);
    }
    else if (data_type == "uint8"){
        Distance<uint8_t>* dist_fn = new DistanceL2UInt8();
        graph_profile<uint8_t>(index_prefix,dist_fn);
    }
    else{
        std::cout<<"Unsupported Data Type."<<std::endl;
    }
}