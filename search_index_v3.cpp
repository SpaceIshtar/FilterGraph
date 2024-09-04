#include "index_v3.h"
#include <chrono>

float calculate_recall(uint32_t nq, uint32_t topk, uint32_t* result, uint32_t* gt_ids, uint32_t gt_dim){
    uint32_t count = 0;
    for (uint32_t i=0;i<nq;i++){
        uint32_t cur_count=0;
        std::vector<uint32_t> result_vec;
        for (uint32_t j=0;j<topk;j++){
            result_vec.push_back(result[i*topk+j]);
        }
        for (uint32_t j=0;j<topk;j++){
            uint32_t id = gt_ids[i*gt_dim+j];
            if (std::find(result_vec.begin(),result_vec.end(),id)!=result_vec.end()){
                count++;
                cur_count++;
            }
        }
    }
    return (float)count/(float)(nq*topk);
}

template<typename T>
void search_queryfile(std::string& query_path, std::string& query_label_file, std::string& index_prefix, std::string& gt_file, 
                        Distance<T>* dist_fn, uint32_t topk, uint32_t ef_search){
    uint32_t nq = 0, dim = 0, topk2 = 0;
    T* query = nullptr;
    read_bin<T>(query_path.c_str(),nq,dim,query);
    uint32_t* gt_ids = nullptr;
    float* gt_dis = nullptr;
    read_gt(gt_file.c_str(),nq,topk2,gt_ids,gt_dis);
    int64_t rows = 0,cols = 0,nnz = 0;
    int64_t* row_index = nullptr;
    int32_t* col_index = nullptr;
    float* value = nullptr;
    if (query_label_file.size()>1){
        read_sparse_matrix(query_label_file.c_str(),rows,cols,nnz,row_index,col_index,value);
    }
    std::vector<std::vector<uint32_t>> query_to_labels(rows,std::vector<uint32_t>());
    #pragma omp parallel for 
    for (uint32_t i=0;i<rows;i++){
        for (uint32_t j=row_index[i];j<row_index[i+1];j++){
            query_to_labels[i].push_back(col_index[j]);
        }
    }
    std::cout<<"Read Data Ends."<<std::endl;
    FilterIndex_v3<T>* index = new FilterIndex_v3<T>(dist_fn,index_prefix);

    std::cout<<"Load Index Ends."<<std::endl;
    uint32_t* result = new uint32_t[nq*topk];
    float* distance_res = new float[nq*topk];
    auto start = std::chrono::high_resolution_clock::now();
    uint32_t computation_count = 0;
    std::mutex count_lock;
    std::vector<uint32_t> query_dis_computation(nq,0);
    // #pragma omp parallel for
    for (uint32_t i=0;i<nq;i++){
        uint32_t cur_count = index->search_predicate(query+i*dim,query_to_labels[i],topk,ef_search,result+i*topk,distance_res+i*topk,false);
        count_lock.lock();
        computation_count+=cur_count;
        count_lock.unlock();
        query_dis_computation[i] = cur_count;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Search Ends."<<std::endl;
    std::chrono::duration<double> diff = end - start;
    double qps = nq/diff.count();
    float recall = calculate_recall(nq,topk,result,gt_ids,topk2);
    double avg_distance_comp = (double)computation_count/(double)nq;
    std::cout<<"QPS: "<<qps<<", recall: "<<recall<<", computation: "<<avg_distance_comp<<std::endl;
}

int main(int argc, char** argv){
    if (argc!=8){
        std::cout<<"usage: ./search_index query_path query_label_spmat index_prefix gt_file topk ef_search data_type"<<std::endl;
        exit(-1);
    }
    std::string query_path = argv[1];
    std::string query_label_file = argv[2];
    std::string index_prefix = argv[3];
    std::string gt_file = argv[4];
    uint32_t topk = atoi(argv[5]);
    uint32_t ef_search = atoi(argv[6]);
    std::string data_type = argv[7];

    if (data_type=="float"){
        Distance<float>* dist_fn = new AVXDistanceL2Float();
        search_queryfile(query_path, query_label_file, index_prefix, gt_file, dist_fn, topk, ef_search);
        delete dist_fn;
    }
    else if (data_type=="uint8"){
        Distance<uint8_t>* dist_fn = new DistanceL2UInt8();
        search_queryfile(query_path, query_label_file, index_prefix, gt_file, dist_fn, topk, ef_search);
        delete dist_fn;
    }
    
    
}