#include <omp.h>

#include "index_sharing.h"
// #include "index_final.h"


template <typename T>
void create_instance(std::string& data_path, std::string& label_file, Distance<T>* dist_fn, uint32_t degree, 
                    uint32_t num_of_entry_points, uint32_t Lsize, uint32_t ef_construct, std::string& save_path, uint32_t pack_k,float r, float threshold_high){
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

    FilterIndex<T>* index = new FilterIndex<T>(nd,dim,degree,dist_fn,num_of_entry_points,pack_k);
    // FilterIndex<T>* index = new FilterIndex<T>(nd,dim,degree,dist_fn,num_of_entry_points,pack_k, r,threshold_high);
    uint32_t count = 0;
    std::mutex count_lock;
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (uint32_t i=0;i<nd;i++){
        index->addPoint(data+i*dim,i,labels[i], Lsize, ef_construct);
        count_lock.lock();
        count++;
        if (count%10000==0){
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout<<"Added "<<count<<" data points. Time spent: "<<diff.count()<<std::endl;
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
    delete[] data;
    delete[] row_index;
    delete[] col_index;
    delete[] values;
}

int main(int argc, char** argv){
    if (argc!=12){
        std::cout<<"usage: ./build_index data_path label_file data_type graph_degree num_of_entry_points Lsize ef_construction pack_k threshold_high save_path"<<std::endl;
        exit(-1);
    }
    std::string data_path = argv[1];
    std::string label_path = argv[2];
    std::string data_type = argv[3];
    uint32_t degree = atoi(argv[4]);
    uint32_t num_of_entry_points = atoi(argv[5]);
    uint32_t Lsize = atoi(argv[6]);
    uint32_t ef = atoi(argv[7]);
    uint32_t pack_k = atoi(argv[8]);
    float r = atof(argv[9]);
    float threshold_high = atoi(argv[10]);
    std::string save_path = argv[11];

    if (data_type=="float"){
        Distance<float>* dist_fn = new AVXDistanceL2Float();
        create_instance<float>(data_path,label_path,dist_fn,degree,num_of_entry_points,Lsize, ef,save_path, pack_k, r, threshold_high);
        delete dist_fn;
    }
    else if (data_type=="uint8"){
        Distance<uint8_t>* dist_fn = new DistanceL2UInt8();
        create_instance<uint8_t>(data_path,label_path,dist_fn,degree,num_of_entry_points,Lsize, ef,save_path, pack_k, r, threshold_high);
        delete dist_fn;
    }
    
}
