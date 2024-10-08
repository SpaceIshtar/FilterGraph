#include <omp.h>

#include "index_v3.h"

template <typename T>
void create_instance(std::string& data_path, std::string& label_file, Distance<T>* dist_fn, uint32_t bigM, 
                    uint32_t num_of_entry_points, uint32_t ef_construct, std::string& save_path){
    uint32_t nd = 0, dim = 0;
    T* data = nullptr;
    read_bin<T>(data_path.c_str(),nd,dim,data);
    int64_t rows = 0, cols = 0, nnz = 0;
    int64_t* row_index=nullptr;
    int32_t* col_index=nullptr;
    float* values=nullptr;
    if (label_file.size()>1){
        read_sparse_matrix(label_file.c_str(),rows,cols,nnz,row_index,col_index,values);
    }
    std::vector<std::vector<uint32_t>> labels(nd,std::vector<uint32_t>());
    for (size_t i=0;i<nd;i++){
        for (size_t j=row_index[i];j<row_index[i+1];j++){
            labels[i].push_back(col_index[j]);
        }
    }

    std::cout<<"Read Data Ends."<<std::endl;
    std::cout<<"nd: "<<nd<<", dim: "<<dim<<std::endl;
    std::cout<<"rows: "<<rows<<"; cols: "<<cols<<"; nnz: "<<nnz<<std::endl;

    FilterIndex_v3<T>* index = new FilterIndex_v3<T>(nd,dim,bigM,dist_fn,num_of_entry_points,1024);
    uint32_t count = 0;
    std::mutex count_lock;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (uint32_t i=0;i<nd;i++){

        index->insert(data+i*dim,i,labels[i], ef_construct);
        count_lock.lock();
        count++;
        if (count%5000==0){
            std::cout<<"Added "<<count<<" data points."<<std::endl;
        }
        count_lock.unlock();

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Build Index Ends"<<std::endl;
    std::chrono::duration<double> diff = end - start;
    std::cout<<"Build time: "<<diff.count()<<std::endl;

    index->save(save_path);
    std::cout<<"Save index Ends"<<std::endl;
    delete index;
    delete data;

}

int main(int argc, char** argv){
    if (argc!=8){
        std::cout<<"usage: ./build_index_v3 data_path label_file data_type degree_budget num_of_entry_points ef_construction save_path"<<std::endl;
        exit(-1);
    }
    std::string data_path = argv[1];
    std::string label_path = argv[2];
    std::string data_type = argv[3];
    uint32_t degree = atoi(argv[4]);
    uint32_t num_of_entry_points = atoi(argv[5]);
    uint32_t ef = atoi(argv[6]);
    std::string save_path = argv[7];

    if (data_type=="float"){
        Distance<float>* dist_fn = new AVXDistanceL2Float();
        create_instance<float>(data_path,label_path,dist_fn,degree,num_of_entry_points, ef,save_path);
        delete dist_fn;
    }
    else if (data_type=="uint8"){
        Distance<uint8_t>* dist_fn = new DistanceL2UInt8();
        create_instance<uint8_t>(data_path,label_path,dist_fn,degree,num_of_entry_points, ef,save_path);
        delete dist_fn;
    }
    
}
