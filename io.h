#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <vector>
#include <algorithm>
#ifndef IO_H
#define IO_H
typedef std::unordered_set<std::string> label_set;
typedef std::string path;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set>, std::unordered_map<std::string, uint32_t>, std::unordered_set<std::string>> parse_label_file_return_values;

template<typename T>
void read_bin(const char* filename, uint32_t& nd, uint32_t& dim, T* &data){
    std::ifstream reader(filename,std::ios::in);
    reader.read((char*)&nd,sizeof(uint32_t));
    reader.read((char*)&dim,sizeof(uint32_t));
    if (data!=nullptr){
        delete[] data;
    }
    std::cout<<nd<<", "<<dim<<std::endl;
    size_t data_size = (size_t)nd*(size_t)dim;
    data = new T[data_size];
    reader.read((char*)data,sizeof(T)*data_size);
    reader.close();
}

parse_label_file_return_values parse_label_file(path label_data_path, std::string universal_label)
{
    std::ifstream label_data_stream(label_data_path);
    std::string line, token;
    uint32_t line_cnt = 0;

    // allows us to reserve space for the points_to_labels vector
    while (std::getline(label_data_stream, line))
        line_cnt++;
    label_data_stream.clear();
    label_data_stream.seekg(0, std::ios::beg);

    // values to return
    std::vector<label_set> point_ids_to_labels(line_cnt);
    std::unordered_map<std::string, uint32_t> labels_to_number_of_points;
    label_set all_labels;

    std::vector<uint32_t> points_with_universal_label;
    line_cnt = 0;
    while (std::getline(label_data_stream, line))
    {
        std::istringstream current_labels_comma_separated(line);
        label_set current_labels;

        // get point id
        uint32_t point_id = line_cnt;

        // parse comma separated labels
        bool current_universal_label_check = false;
        while (getline(current_labels_comma_separated, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

            // if token is empty, there's no labels for the point
            if (token == universal_label)
            {
                points_with_universal_label.push_back(point_id);
                current_universal_label_check = true;
            }
            else
            {
                all_labels.insert(token);
                current_labels.insert(token);
                labels_to_number_of_points[token]++;
            }
        }

        if (current_labels.size() <= 0 && !current_universal_label_check)
        {
            std::cerr << "Error: " << point_id << " has no labels." << std::endl;
            exit(-1);
        }
        point_ids_to_labels[point_id] = current_labels;
        line_cnt++;
    }

    // for every point with universal label, set its label set to all labels
    // also, increment the count for number of points a label has
    for (const auto &point_id : points_with_universal_label)
    {
        point_ids_to_labels[point_id] = all_labels;
        for (const auto &lbl : all_labels)
            labels_to_number_of_points[lbl]++;
    }

    std::cout << "Identified " << all_labels.size() << " distinct label(s) for " << point_ids_to_labels.size()
              << " points\n"
              << std::endl;

    return std::make_tuple(point_ids_to_labels, labels_to_number_of_points, all_labels);
}

void read_sparse_matrix(const char* filename,int64_t& rows, int64_t& cols, int64_t& nnz, int64_t*& row_index, int32_t*& col_index, float*& value){
    std::ifstream reader(filename,std::ios::binary|std::ios::in);
    reader.read((char*)&rows,sizeof(int64_t));
    reader.read((char*)&cols,sizeof(int64_t));
    reader.read((char*)&nnz,sizeof(int64_t));
    std::cout<<"Matrix size: ("<<rows<<", "<<cols<<"), non-zeros elements: "<<nnz<<std::endl;
    if (row_index!=nullptr){
        delete[] row_index;
    }
    row_index = new int64_t[rows+1];
    if (col_index!=nullptr){
        delete[] col_index;
    }
    if (value!=nullptr){
        delete[] value;
    }
    col_index = new int32_t[nnz];
    value = new float[nnz];
    reader.read((char*)row_index,sizeof(int64_t)*(rows+1));
    reader.read((char*)col_index,sizeof(int32_t)*nnz);
    reader.read((char*)value,sizeof(float)*nnz);
    reader.close();
}

void write_sparse_matrix(const char* filename, int64_t rows, int64_t cols, int64_t nnz, int64_t* row_index, int32_t* col_index, float* value){
    std::ofstream writer(filename);
    writer.write((char*)&rows,sizeof(int64_t));
    writer.write((char*)&cols,sizeof(int64_t));
    writer.write((char*)&nnz,sizeof(int64_t));
    writer.write((char*)row_index,sizeof(int64_t)*(rows+1));
    writer.write((char*)col_index,sizeof(int32_t)*nnz);
    writer.write((char*)value,sizeof(float)*nnz);
    writer.close();
}

template<typename T>
void write_bin(const char* filename,uint32_t nd, uint32_t dim, T* data){
    std::ofstream writer(filename,std::ios::out);
    writer.write((char*)&nd,sizeof(uint32_t));
    writer.write((char*)&dim,sizeof(uint32_t));
    writer.write((char*)data,sizeof(T)*nd*dim);
    writer.close();
}

void read_gt(const char* filename, uint32_t& nq, uint32_t& topk, uint32_t*& gt_ids, float*& gt_dis){
    std::ifstream reader(filename);
    reader.read((char*)&nq,sizeof(uint32_t));
    reader.read((char*)&topk,sizeof(uint32_t));
    if (gt_ids!=nullptr){
        delete[] gt_ids;
    }
    if (gt_dis!=nullptr){
        delete[] gt_dis;
    }
    gt_ids = new uint32_t[nq*topk];
    gt_dis = new float[nq*topk];
    reader.read((char*)gt_ids,sizeof(uint32_t)*nq*topk);
    reader.read((char*)gt_dis,sizeof(float)*nq*topk);
    reader.close();
}

void write_gt(const char* filename, uint32_t nq, uint32_t topk, uint32_t* gt_ids, float* gt_dis){
    std::ofstream writer(filename);
    writer.write((char*)&nq,sizeof(uint32_t));
    writer.write((char*)&topk,sizeof(uint32_t));
    writer.write((char*)gt_ids,sizeof(uint32_t)*nq*topk);
    writer.write((char*)gt_dis,sizeof(float)*nq*topk);
    writer.close();
}

void read_ivecs_gt(const char* filename, uint32_t& num, uint32_t& dim, uint32_t*& data){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    in.read((char*)&dim,4);
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    data = new uint32_t[num * dim * sizeof(float)];

    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

#endif