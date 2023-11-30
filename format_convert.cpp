#include "io.h"
#include <cstring>

int main(int argc, char** argv){
    if (argc!=4){
        std::cout<<"usage: ./format_convert origin_file save_path conversion_type"<<std::endl;
        std::cout<<"type 0: convert ivecs groundtruth to bin file"<<std::endl;
        exit(-1);
    }

    std::string file_path = argv[1];
    std::string save_path = argv[2];
    int type = atoi(argv[3]);

    if (type == 0){
        uint32_t nq, topk;
        uint32_t* gt_ids = nullptr;
        read_ivecs_gt(file_path.c_str(),nq,topk,gt_ids);
        float* gt_dis = new float[nq*topk];
        memset(gt_dis,0,sizeof(float)*nq*topk);
        write_gt(save_path.c_str(),nq,topk,gt_ids,gt_dis);
    }
    else{
        std::cout<<"Unsupported Conversion Type."<<std::endl;
    }
}