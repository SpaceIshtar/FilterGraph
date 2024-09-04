#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <mutex>
#include <queue>
#include <cmath>
#include <algorithm>
#include <set>
#include <omp.h>

#include "distance.h"
#include "io.h"

#define MAX_LOCKS 65536
#define getLockIndex(a) a&65535
#define LARGE_TAG 0
#define CARDINALITY_THRESHOLD_LOW 1024

template<typename T>
class FilterIndex{
    public:
        T* _vec_store = nullptr;
        uint32_t* _graph_store = nullptr;
        size_t _dim = 0;
        size_t _nd = 0;
        size_t _graph_degree = 0;
        size_t _pack_k = 0;
        Distance<T>* _dist_fn = nullptr;
        uint32_t _num_of_entry_point = 5;
        uint32_t curBlockID = 0;
        float _r = 0;
        float CARDINALITY_THRESHOLD_HIGH = 200000;

        std::vector<std::vector<uint32_t>> _pts_to_labels;
        std::unordered_map<uint32_t,std::vector<uint32_t>> _label_to_pts;
        std::unordered_map<uint32_t,std::vector<uint32_t>> _label_to_medoid_id;
        std::unordered_map<uint32_t, bool> _label_graph_check;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> _pts_tag_index;
        std::vector<std::pair<uint32_t,uint32_t>> _pts_graph_index;
        std::unordered_map<uint32_t, float> _label_hierarchy_level;
        std::vector<std::vector<uint32_t>> _block_to_pts;
        std::vector<uint32_t> _pack_buffer;
        mutable std::vector<std::mutex> _label_lock;
        std::mutex _global_mtx;

        FilterIndex(Distance<T>* dist_fn){
            _dist_fn = dist_fn;
            _label_lock = std::vector<std::mutex>(MAX_LOCKS);
        }

        FilterIndex(size_t nd, size_t dim, size_t graph_degree, Distance<T>* dist_fn, uint32_t num_of_entry_point = 5, uint32_t pack_k = 10, float r = 0.15f, float CARDINALITY_THRESHOLD_HIGH_ = 200000){
            _dim = dim;
            _nd = nd;
            _graph_degree = graph_degree;
            _vec_store = (T*)malloc(sizeof(T)*nd*dim);

            _graph_store = (uint32_t*)malloc(sizeof(uint32_t)*nd*graph_degree);
            memset(_graph_store, -1, sizeof(uint32_t)*_nd*_graph_degree);
            _pts_to_labels.resize(nd);
            _pts_graph_index = std::vector<std::pair<uint32_t,uint32_t>>(nd, std::pair<uint32_t,uint32_t>(std::numeric_limits<uint32_t>::max(),-1));
            _pts_tag_index.resize(nd);
            _label_lock = std::vector<std::mutex>(MAX_LOCKS);
            _dist_fn = dist_fn;
            _num_of_entry_point = num_of_entry_point;
            _pack_k = pack_k;
            curBlockID = 0;
            _r = r;
            CARDINALITY_THRESHOLD_HIGH = CARDINALITY_THRESHOLD_HIGH_;
        }

        ~FilterIndex(){
            free(_vec_store);
            free(_graph_store);
        }

        bool match_all_filters(uint32_t point_id, const std::vector<uint32_t> &incoming_labels){
            auto &curr_node_labels = _pts_to_labels[point_id];
            auto cur_pointer = curr_node_labels.begin();
            for (uint32_t label:incoming_labels){
                cur_pointer = std::find(cur_pointer,curr_node_labels.end(),label);
                if (cur_pointer==curr_node_labels.end()) return false;
            }
            return true;
        }

        float getLevel(float card){
            return std::log(card)/std::log(_r);
        }

        void search_graph_OR(T* query, std::vector<uint32_t>& labels, std::set<uint32_t> start_points, uint32_t ef_search, std::priority_queue<std::pair<float, uint32_t>>& pq, bool verbose = false){
            bool small_tag = true;
            _global_mtx.lock();
            std::unordered_map<uint32_t, float> hierarchy_level_copy = _label_hierarchy_level;
            _global_mtx.unlock();
            for (uint32_t tag: labels){
                if (hierarchy_level_copy[tag] >= getLevel(CARDINALITY_THRESHOLD_HIGH)){
                    small_tag = false;
                }
                else if (!small_tag){
                    std::cerr<<"Labels should be all large tags or all medium tags"<<std::endl;
                }
            }
            std::priority_queue<std::pair<float,uint32_t>> ef_queue;
            std::vector<uint32_t> visit_set;
            for (uint32_t entry_id:start_points){
                if (entry_id >= _nd) break;
                float dis = _dist_fn->compare(_vec_store+entry_id*_dim,query,_dim);
                ef_queue.emplace(-dis,entry_id);
                visit_set.push_back(entry_id);
                pq.emplace(dis,entry_id);
            }
            float lower_bound = std::numeric_limits<float>::max();
            if (!pq.empty()) lower_bound = pq.top().first;
            std::vector<uint32_t> candidates;
            while (!ef_queue.empty()){
                std::pair<float,uint32_t> cur = ef_queue.top();
                float dis = -cur.first;
                uint32_t id = cur.second;
                if (dis>lower_bound && pq.size()>=ef_search){
                    break;
                }
                ef_queue.pop();

                candidates.clear();
                uint32_t* start_pointer = _graph_store + _pts_graph_index[id].first;
                if (small_tag){
                    for (uint32_t i = 0; i < _pts_tag_index[id].size() - 1; i++){
                        uint32_t tag = _pts_tag_index[id][i].first;
                        if (std::find(labels.begin(),labels.end(),tag)==labels.end()) continue;
                        uint32_t start_index = _pts_tag_index[id][i].second;
                        uint32_t end_index = _pts_tag_index[id][i+1].second;
                        for (uint32_t* pointer = start_pointer + start_index; pointer < start_pointer + end_index; ++pointer){
                            uint32_t neighbor = *pointer;
                            if (neighbor >= _nd) break;
                            if (std::find(visit_set.begin(),visit_set.end(),neighbor) == visit_set.end()){
                                visit_set.push_back(neighbor);
                                candidates.push_back(neighbor);
                            }
                        }
                    }
                }
                else{
                    for (uint32_t i = 0; i < _pts_tag_index[id].size() - 1; i++){
                        uint32_t tag = _pts_tag_index[id][i].first;
                        if (tag != LARGE_TAG && std::find(labels.begin(),labels.end(),tag)==labels.end()) continue;
                        uint32_t start_index = _pts_tag_index[id][i].second;
                        uint32_t end_index = _pts_tag_index[id][i+1].second;
                        for (uint32_t* pointer = start_pointer + start_index; pointer < start_pointer + end_index; ++pointer){
                            uint32_t neighbor = *pointer;
                            if (neighbor >= _nd) break;
                            if (std::find(visit_set.begin(),visit_set.end(),neighbor) == visit_set.end()){
                                visit_set.push_back(neighbor);
                                candidates.push_back(neighbor);
                            }
                        }
                    }
                }

                for (int i=0;i<candidates.size();i++){
                    if (i<candidates.size()-1){
                        _mm_prefetch(_vec_store+candidates[i+1]*_dim,_MM_HINT_T0);
                    }
                    float candidate_dis = _dist_fn->compare(_vec_store+candidates[i]*_dim,query,_dim);
                    if (pq.size()<ef_search||lower_bound>candidate_dis){
                        ef_queue.emplace(-candidate_dis,candidates[i]);
                        pq.emplace(candidate_dis,candidates[i]);
                        if (pq.size()>ef_search){
                            pq.pop();
                        }
                        if (!pq.empty()){
                            lower_bound=pq.top().first;
                        }
                    }
                }

            }
        }

        void traverse_neighbor(uint32_t id, uint32_t tag, std::vector<uint32_t>& visit_set, std::vector<uint32_t>& candidates, bool verbose = false){
            candidates.clear();
            _global_mtx.lock();
            uint32_t tag_cardinality = _label_to_pts[tag].size();
            _global_mtx.unlock();
            if (tag_cardinality < CARDINALITY_THRESHOLD_HIGH){
                uint32_t start_index = 0, end_index = 0;
                for (int i=0;i<_pts_tag_index[id].size() - 1;i++){
                    // if (_pts_tag_index[id].empty()) break;
                    std::pair<uint32_t,uint32_t>& row_index = _pts_tag_index[id][i];
                    if (row_index.first == tag){
                        start_index = row_index.second;
                        end_index = _pts_tag_index[id][i+1].second;
                        break;
                    }
                }
                if (_pts_graph_index[id].first == std::numeric_limits<uint32_t>::max()) return;
                uint32_t* start_pointer = _graph_store + _pts_graph_index[id].first;
                uint32_t* end_pointer = start_pointer+end_index;
                start_pointer+=start_index;
                if (verbose){
                    std::cout<<"Search centroid: "<<id<<", target filter: "<<tag<<", allocated degree: "<<(end_pointer - start_pointer)<<std::endl;
                }
                for (uint32_t* neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                    uint32_t neighbor_id=*neighbor_pointer;
                    if (neighbor_id>_nd) break;
                    if (std::find(visit_set.begin(),visit_set.end(),neighbor_id)==visit_set.end()){
                        visit_set.push_back(neighbor_id);
                        candidates.push_back(neighbor_id);
                    }
                }
            }
            else{
                uint32_t start_index = 0, end_index = 0;
                for (int i=0;i<_pts_tag_index[id].size() -1;i++){
                    std::pair<uint32_t,uint32_t>& row_index = _pts_tag_index[id][i];
                    if (row_index.first == LARGE_TAG || row_index.first == tag){
                        start_index = row_index.second;
                        end_index = _pts_tag_index[id][i+1].second;
                        break;
                    }
                }
                if (_pts_graph_index[id].first == std::numeric_limits<uint32_t>::max()) return;
                uint32_t* start_pointer = _graph_store + _pts_graph_index[id].first;
                uint32_t* end_pointer = start_pointer+end_index;
                start_pointer+=start_index;
                if (verbose){
                    std::cout<<"Search centroid: "<<id<<", target filter: "<<tag<<", allocated degree: "<<(end_pointer - start_pointer)<<std::endl;
                }
                for (uint32_t* neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                    uint32_t neighbor_id=*neighbor_pointer;
                    if (neighbor_id>_nd) break;
                    if (std::find(_pts_to_labels[neighbor_id].begin(),_pts_to_labels[neighbor_id].end(),tag)==_pts_to_labels[neighbor_id].end()) continue;
                    if (std::find(visit_set.begin(),visit_set.end(),neighbor_id)==visit_set.end()){
                        visit_set.push_back(neighbor_id);
                        candidates.push_back(neighbor_id);
                    }
                }
            }
            
            return;
        }

        uint32_t search(T* query, std::vector<uint32_t> labels, uint32_t topk, uint32_t* ids_res, float* dis_res, uint32_t ef_search, bool verbose=false){
            // cardinality estimation
            uint32_t cardinality = _nd;
            uint32_t smallest_cardinality = _nd;
            if (labels.empty()){
                labels.emplace_back(std::numeric_limits<uint32_t>::max());
            }
            std::sort(labels.begin(),labels.end());
            uint32_t smallest_filter = labels[0];
            uint32_t computation_count = 0;
            if (verbose){
                std::cout<<"labels: ";
                for (uint32_t label:labels){
                    std::cout<<label<<",";
                }
                std::cout<<std::endl;
            }
            bool smallest_graph_search = true;
            {
                std::vector<float> label_size;
                for (uint32_t label:labels){
                    _global_mtx.lock();
                    label_size.push_back(_label_to_pts[label].size());
                    if (_label_to_pts[label].size()<smallest_cardinality){
                        smallest_cardinality = _label_to_pts[label].size();
                        smallest_filter = label;
                        smallest_graph_search = _label_graph_check[label];
                    }
                    _global_mtx.unlock();
                }
                std::sort(label_size.begin(),label_size.end());
                for (int i=0;i<label_size.size();i++){
                    float fraction = label_size[i]/(float)_nd;
                    for (uint32_t j=0;j<i;j++){
                        fraction = std::sqrt(fraction);
                    }
                    cardinality = cardinality*fraction;
                }
            }
            // query plan selection: whether search graph, traverse which filter
            // if brute-force
            std::priority_queue<std::pair<float,uint32_t>> pq;
            if (verbose){
                std::cout<<"cardinality: "<<cardinality<<", smallest cardinality: "<<smallest_cardinality<<", smallest label: "<<smallest_filter<<", graph_search: "<<smallest_graph_search<<std::endl;
            }
            // bool use_graph = (cardinality >= 1024 || smallest_cardinality >= 4096) && smallest_graph_search;
            // if (!use_graph){
            if (cardinality < 1000 || smallest_cardinality < 1000 || !smallest_graph_search){
                // go through smallest filter
                std::vector<uint32_t> candidates;
                _global_mtx.lock();
                std::vector<uint32_t> all_candidates(_label_to_pts[smallest_filter]);
                _global_mtx.unlock();
                for (uint32_t point_id:all_candidates){
                    if (match_all_filters(point_id,labels)){
                        candidates.push_back(point_id);
                    }
                }
                for (int i=0;i<candidates.size();i++){
                    if (i<candidates.size()-1){
                        _mm_prefetch(_vec_store+candidates[i+1]*_dim,_MM_HINT_T0);
                    }
                    float dis = _dist_fn->compare(_vec_store+candidates[i]*_dim,query,_dim);
                    if (verbose){
                        std::cout<<"BF: candidate_id = "<<candidates[i]<<", dis = "<<dis<<std::endl;
                    }
                    computation_count++;
                    pq.emplace(dis,candidates[i]);
                }
            }
            // if search graph
            else{
                std::priority_queue<std::pair<float,uint32_t>> ef_queue;
                std::priority_queue<std::pair<float,uint32_t>> top_candidates;
                std::vector<uint32_t> visit_set;
                _global_mtx.lock();
                std::vector<uint32_t> start_points(_label_to_medoid_id[smallest_filter]);
                _global_mtx.unlock();
                for (uint32_t entry_id:start_points){
                    float dis = _dist_fn->compare(_vec_store+entry_id*_dim,query,_dim);
                    computation_count++;
                    ef_queue.emplace(-dis,entry_id);
                    top_candidates.emplace(dis,entry_id);
                    visit_set.push_back(entry_id);
                    if (match_all_filters(entry_id,labels)){
                        pq.emplace(dis,entry_id);
                        if (verbose){
                            std::cout<<"match: candidate="<<entry_id<<", dis="<<dis<<std::endl;
                        }
                    }
                    else if (verbose){
                        std::cout<<"traverse: candidiate="<<entry_id<<", dis="<<dis<<std::endl;
                    }
                }
                float lower_bound = std::numeric_limits<float>::max();
                if (!top_candidates.empty()) lower_bound = top_candidates.top().first;
                std::vector<uint32_t> candidates;
                while (!ef_queue.empty()){
                    std::pair<float,uint32_t> cur = ef_queue.top();
                    float dis = -cur.first;
                    uint32_t id = cur.second;
                    if (dis>lower_bound && top_candidates.size()>=ef_search){
                        break;
                    }
                    ef_queue.pop();

                    traverse_neighbor(id, smallest_filter, visit_set, candidates, verbose);

                    for (int i=0;i<candidates.size();i++){
                        if (i<candidates.size()-1){
                            _mm_prefetch(_vec_store+candidates[i+1]*_dim,_MM_HINT_T0);
                        }
                        float candidate_dis = _dist_fn->compare(_vec_store+candidates[i]*_dim,query,_dim);
                        bool match_filter = match_all_filters(candidates[i],labels);
                        if (match_filter){
                            pq.emplace(candidate_dis,candidates[i]);
                            if (verbose){
                                std::cout<<"match: candidate="<<candidates[i]<<", dis="<<candidate_dis<<std::endl;
                            }
                        }
                        computation_count++;
                        if (top_candidates.size()<ef_search||lower_bound>candidate_dis){
                            ef_queue.emplace(-candidate_dis,candidates[i]);
                            top_candidates.emplace(candidate_dis,candidates[i]);
                            if (verbose && !match_filter){
                                std::cout<<"traverse: candidiate="<<candidates[i]<<", dis="<<candidate_dis<<std::endl;
                            }
                            if (top_candidates.size()>ef_search){
                                top_candidates.pop();
                            }
                            if (!top_candidates.empty()){
                                lower_bound=top_candidates.top().first;
                            }
                        }
                        else if (verbose){
                            std::cout<<"candidate further than lower bound: candidate="<<candidates[i]<<", dis="<<candidate_dis<<", lower bound: "<<lower_bound<<std::endl;
                        }
                    }

                }
            }

            // fetch result from priority queue
            while (pq.size()>topk){
                pq.pop();
            }

            uint32_t position = pq.size();
            while (!pq.empty()){
                position--;
                std::pair<float,uint32_t> p = pq.top();
                pq.pop();
                ids_res[position] = p.second;
                dis_res[position] = p.first;
            }
            return computation_count;

        }

        void pruning(uint32_t target, std::priority_queue<std::pair<float,uint32_t>>& pq, std::vector<uint32_t>& target_tags, std::vector<uint32_t>& pruning_result, uint32_t degree, float alpha){
            std::sort(target_tags.begin(),target_tags.end());
            std::priority_queue<std::pair<float, uint32_t>> small_pq;
            while (!pq.empty()){
                auto p = pq.top();
                small_pq.emplace(-p.first, p.second);
                pq.pop();
            }
            while (!small_pq.empty()){
                auto p = small_pq.top();
                small_pq.pop();
                float dis1 = -p.first;
                uint32_t cand = p.second;
                if (cand == target) continue;
                bool prune = false;
                std::vector<uint32_t> cand_target;
                std::set_intersection(_pts_to_labels[cand].begin(),_pts_to_labels[cand].end(),target_tags.begin(),target_tags.end(),std::back_inserter(cand_target));
                for (uint32_t exist_neigh: pruning_result){
                    std::vector<uint32_t> exist_neigh_target;
                    bool tag_dominate = false;
                    std::set_intersection(_pts_to_labels[exist_neigh].begin(),_pts_to_labels[exist_neigh].end(),target_tags.begin(),target_tags.end(),std::back_inserter(exist_neigh_target));
                    if (std::includes(exist_neigh_target.begin(),exist_neigh_target.end(),cand_target.begin(),cand_target.end())){
                        tag_dominate = true;
                    }
                    if (tag_dominate){
                        float dis2 = _dist_fn->compare((_vec_store+exist_neigh*_dim),(_vec_store+cand*_dim),_dim);
                        if (dis1 > dis2 * alpha){
                            prune = true;
                            break;
                        }
                    }
                }
                if (!prune){
                    pruning_result.push_back(cand);
                    if (pruning_result.size()>=degree) break;
                }
            }
        }

        void addPoint(T* data, size_t id, std::vector<uint32_t>& tags, uint32_t Lsize = 100, uint32_t ef_construction=100, float alpha = 1.2f){
            if (tags.empty()){
                tags.emplace_back(std::numeric_limits<uint32_t>::max());
            }
            if (std::find(tags.begin(),tags.end(),LARGE_TAG)!=tags.end()){
                std::cout<<"Tag 0 is preserved for large tags"<<std::endl;
                exit(-1);
            }
            std::sort(tags.begin(),tags.end());
            {
                uint32_t lock_id = getLockIndex(id);
                _label_lock[lock_id].lock();
                memcpy(_vec_store+id*_dim,data,_dim*sizeof(T));     
                _label_lock[lock_id].unlock();
            }
            std::vector<uint32_t> todolist;
            uint32_t allocate_start_point = std::numeric_limits<uint32_t>::max();
            // insert into _label_to_pts update _pts_to_lables
            _global_mtx.lock();
            for (uint32_t label:tags){
                if (_label_to_pts.find(label)==_label_to_pts.end()){
                    _label_to_pts[label] = std::vector<uint32_t>(1,id);
                    _label_graph_check[label] = false;
                }
                else{
                   _label_to_pts[label].push_back(id); 
                }
            }
            _pts_to_labels[id] = tags;
            _pack_buffer.push_back(id);
            uint32_t current_block_id = curBlockID;
            if (_pack_buffer.size()==_pack_k){
                todolist.swap(_pack_buffer);
                allocate_start_point = curBlockID * _pack_k * _graph_degree;
                curBlockID++;
                _block_to_pts.push_back(todolist);
            }
            
            _global_mtx.unlock();
            if (todolist.size()==0) return;
            if (todolist.size()!=_pack_k){
                std::cout<<"todolist size should be equal to _pack_k"<<std::endl;
                exit(-1);
            }
            // allocate for this block
            float level_sum = 0;
            std::vector<uint32_t> labels_need_graph_build;
            std::vector<uint32_t> labels_need_merge;
            _global_mtx.lock();
            for (uint32_t todo_id: todolist){
                float maximum_level = 0.0f;
                for (uint32_t tag: _pts_to_labels[todo_id]){
                    float cur_level_for_label = std::max(1.0f, getLevel(_label_to_pts[tag].size()));
                    if (_label_hierarchy_level[tag] < getLevel(CARDINALITY_THRESHOLD_HIGH) && cur_level_for_label>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                        labels_need_merge.emplace_back(tag);
                    }
                    if (_label_hierarchy_level[tag] < getLevel(CARDINALITY_THRESHOLD_LOW) && cur_level_for_label>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                        labels_need_graph_build.emplace_back(tag);
                    }
                    if (cur_level_for_label>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                        maximum_level += cur_level_for_label;
                    }
                    else if (cur_level_for_label>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                        level_sum += cur_level_for_label;
                    }
                    _label_hierarchy_level[tag] = cur_level_for_label;
                }
                level_sum += maximum_level;
            }
            uint32_t current_position = allocate_start_point;
            for (uint32_t todo_id: todolist){
                uint32_t start_position = current_position;
                uint32_t tag_current_position = 0;
                uint32_t large_degree = 0;
                bool exist_large_degree = false;
                float maximal_level = 0.0f;
                for (uint32_t tag: _pts_to_labels[todo_id]){
                    if (_label_hierarchy_level[tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                        maximal_level += _label_hierarchy_level[tag];
                        exist_large_degree = true;
                    }
                }
                if (exist_large_degree){
                    large_degree = (float)maximal_level/(float)level_sum*_graph_degree*_pack_k;
                    _pts_tag_index[todo_id].push_back(std::pair<uint32_t, uint32_t>(LARGE_TAG, tag_current_position));
                    tag_current_position += large_degree;
                }
                for (uint32_t tag:_pts_to_labels[todo_id]){
                    if (_label_hierarchy_level[tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                        continue;
                    }
                    _pts_tag_index[todo_id].push_back(std::pair<uint32_t,uint32_t>(tag,tag_current_position));
                    if (_label_hierarchy_level[tag]>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                        uint32_t degree = (float)_label_hierarchy_level[tag]/(float)level_sum*_graph_degree*_pack_k;
                        tag_current_position += degree;
                    }
                }
                _pts_tag_index[todo_id].push_back(std::pair<uint32_t,uint32_t>(-1,tag_current_position));
                current_position = start_position + tag_current_position;
                _pts_graph_index[todo_id] = std::pair<uint32_t,uint32_t>(start_position,current_position);
            }
            std::unordered_map<uint32_t, float> hierarchy_level_copy = _label_hierarchy_level;
            _global_mtx.unlock();

            for (uint32_t merging_tag: labels_need_merge){
                // merge tag to LARGE_TAG
                _global_mtx.lock();
                std::vector<uint32_t> tag_points(_label_to_pts[merging_tag]);
                _global_mtx.unlock();
                std::vector<uint32_t> blocks_need_reallocate;
                for (uint32_t point: tag_points){
                    if (_pts_graph_index[point].first != std::numeric_limits<uint32_t>::max()){
                        uint32_t block_id = _pts_graph_index[point].first/(_pack_k*_graph_degree);
                        if (std::find(blocks_need_reallocate.begin(),blocks_need_reallocate.end(),block_id)==blocks_need_reallocate.end()){
                            blocks_need_reallocate.push_back(block_id);
                        }
                    }
                }
                for (uint32_t block: blocks_need_reallocate){
                    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> new_graph_degree;
                    std::unordered_map<uint32_t, std::pair<uint32_t,uint32_t>> new_graph_index;
                    std::unordered_map<uint32_t, std::vector<uint32_t>> new_neighbors;
                    float block_level_sum = 0;
                    _global_mtx.lock();
                    std::vector<uint32_t> block_pts(_block_to_pts[block]);
                    hierarchy_level_copy = _label_hierarchy_level;
                    _global_mtx.unlock();
                    for (uint32_t point: block_pts){
                        new_graph_degree[point] = std::vector<std::pair<uint32_t, uint32_t>>();
                        float maximum_level = 0.0f;
                        for (uint32_t tag: _pts_to_labels[point]){
                            if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                maximum_level += hierarchy_level_copy[tag];
                            }
                            else if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                                block_level_sum += hierarchy_level_copy[tag];
                            }
                        }
                        block_level_sum += maximum_level;
                    }

                    uint32_t block_start_point = block * _pack_k * _graph_degree;
                    for (uint32_t point: block_pts){
                        uint32_t point_lock_id = getLockIndex(point);
                        _label_lock[point_lock_id].lock();
                        uint32_t tag_current_position = 0;
                        uint32_t large_degree = 0;
                        std::vector<uint32_t> large_tags;
                        std::set<uint32_t> large_tag_neighbors;
                        float maximum_level = 0.0f;
                        bool exist_large_degree = false;
                        uint32_t* prev_neighbor_list = _graph_store + _pts_graph_index[point].first;
                        for (int i = 0; i < _pts_to_labels[point].size(); i++){
                            uint32_t tag = _pts_to_labels[point][i];
                            if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                exist_large_degree = true;
                                large_tags.push_back(tag);
                                maximum_level += hierarchy_level_copy[tag];
                            }
                        }
                        if (exist_large_degree){
                            large_degree = (float)maximum_level/(float)block_level_sum*_graph_degree*_pack_k;
                            new_graph_degree[point].push_back(std::pair<uint32_t, uint32_t>(LARGE_TAG, tag_current_position));
                            tag_current_position += large_degree;
                        }
                        for (int i = 0; i < _pts_tag_index[point].size() - 1; i++){
                            uint32_t tag = _pts_tag_index[point][i].first;
                            if (tag == LARGE_TAG || std::find(large_tags.begin(),large_tags.end(),tag)!=large_tags.end()){
                                uint32_t start_index = _pts_tag_index[point][i].second;
                                uint32_t end_index = _pts_tag_index[point][i+1].second;
                                large_tag_neighbors.insert(prev_neighbor_list+start_index, prev_neighbor_list+end_index);
                                continue;
                            }
                            new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(tag,tag_current_position));
                            if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                                uint32_t cur_degree = (float)hierarchy_level_copy[tag]/(float)block_level_sum*_graph_degree*_pack_k;
                                tag_current_position += cur_degree;
                            }
                        }
                        new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(-1,tag_current_position));
                        uint32_t end_position = block_start_point + tag_current_position;
                        new_graph_index[point] = std::pair<uint32_t,uint32_t>(block_start_point, end_position);
                        block_start_point = end_position;
                        uint32_t point_new_total_degree = new_graph_index[point].second - new_graph_index[point].first;
                        new_neighbors[point] = std::vector<uint32_t>(point_new_total_degree,std::numeric_limits<uint32_t>::max());
                        if (_pts_graph_index[point].first==std::numeric_limits<uint32_t>::max()){
                            std::cout<<"The start index has not been defined"<<std::endl;
                            exit(-1);
                        }
                        for (int i = 0; i < new_graph_degree[point].size()-1; i++){
                            uint32_t start_index = new_graph_degree[point][i].second;
                            uint32_t end_index = new_graph_degree[point][i+1].second;
                            uint32_t tag = new_graph_degree[point][i].first;
                            uint32_t degree = end_index - start_index;
                            if (degree == 0) continue;
                            if (tag == LARGE_TAG){
                                large_tag_neighbors.erase(std::numeric_limits<uint32_t>::max());
                                if (large_tag_neighbors.size() <= degree){
                                    std::vector<uint32_t> neighbors(large_tag_neighbors.begin(),large_tag_neighbors.end());
                                    memcpy(new_neighbors[point].data()+start_index, neighbors.data(),sizeof(uint32_t)*neighbors.size());
                                }
                                else{
                                    std::priority_queue<std::pair<float, uint32_t>> pq;
                                    for (uint32_t cand: large_tag_neighbors){
                                        if (cand >= _nd) continue;
                                        float dis = _dist_fn->compare((_vec_store+point*_dim),(_vec_store+cand*_dim),_dim);
                                        pq.emplace(dis, cand);
                                    }
                                    std::vector<uint32_t> neighbors;
                                    pruning(point, pq ,large_tags,neighbors,degree,alpha);
                                    

                                    memcpy(new_neighbors[point].data()+start_index, neighbors.data(),sizeof(uint32_t)*neighbors.size());
                                }
                                
                                continue;
                            }

                            uint32_t prev_start_index = 0;
                            uint32_t prev_end_index = 0;
                            for (uint32_t j = 0; j < _pts_tag_index[point].size() - 1; j++){
                                if (tag == _pts_tag_index[point][j].first){
                                    prev_start_index = _pts_tag_index[point][j].second;
                                    prev_end_index = _pts_tag_index[point][j+1].second;
                                    break;
                                }
                            }
                            uint32_t prev_degree = prev_end_index - prev_start_index;
                            
                            // copy from previous neighbors
                            if (degree >= prev_degree && prev_degree > 0){
                                memcpy(new_neighbors[point].data()+start_index, prev_neighbor_list+prev_start_index,prev_degree*sizeof(uint32_t));
                            }
                            if (degree < prev_degree){
                                memcpy(new_neighbors[point].data()+start_index, prev_neighbor_list+prev_start_index,degree*sizeof(uint32_t));
                            }
                            
                        }
                        _label_lock[point_lock_id].unlock();
                    }
                    for (uint32_t point: block_pts){
                        uint32_t point_lock_id = getLockIndex(point);
                        _label_lock[point_lock_id].lock();
                        _pts_tag_index[point] = new_graph_degree[point];
                        _pts_graph_index[point] = new_graph_index[point];
                        memcpy(_graph_store + new_graph_index[point].first, new_neighbors[point].data(), new_neighbors[point].size()*sizeof(uint32_t));
                        _label_lock[point_lock_id].unlock();
                    }
                }
            }
            
            {
                for (uint32_t building_tag: labels_need_graph_build){
                    // reallocate neighbor for all blocks containing for this tag
                    std::vector<uint32_t> blocks_need_reallocate;
                    _global_mtx.lock();
                    for (uint32_t point:_label_to_pts[building_tag]){
                        if (_pts_graph_index[point].first != std::numeric_limits<uint32_t>::max()){
                            uint32_t block_id = _pts_graph_index[point].first/(_pack_k*_graph_degree);
                            if (std::find(blocks_need_reallocate.begin(),blocks_need_reallocate.end(),block_id)==blocks_need_reallocate.end()){
                                blocks_need_reallocate.push_back(block_id);
                            }
                        }
                    }
                    _global_mtx.unlock();
                    for (uint32_t block: blocks_need_reallocate){
                        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> new_graph_degree;
                        std::unordered_map<uint32_t, std::pair<uint32_t,uint32_t>> new_graph_index;
                        std::unordered_map<uint32_t, std::vector<uint32_t>> new_neighbors;
                        float block_level_sum = 0;
                        _global_mtx.lock();
                        std::vector<uint32_t> block_pts(_block_to_pts[block]);
                        hierarchy_level_copy = _label_hierarchy_level;
                        _global_mtx.unlock();
                        for (uint32_t point: block_pts){
                            new_graph_degree[point] = std::vector<std::pair<uint32_t, uint32_t>>();
                            float maximum_level = 0.0f;
                            for (uint32_t tag: _pts_to_labels[point]){
                                if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                    maximum_level += hierarchy_level_copy[tag];
                                }
                                else if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                                    block_level_sum += hierarchy_level_copy[tag];
                                }
                            }
                            block_level_sum += maximum_level;
                        }
                        uint32_t block_start_point = block * _pack_k * _graph_degree;
                        for (uint32_t point: block_pts){
                            uint32_t point_lock_id = getLockIndex(point);
                            _label_lock[point_lock_id].lock();
                            uint32_t tag_current_position = 0;
                            uint32_t large_degree = 0;
                            std::vector<uint32_t> large_tags;

                            for (int i = 0; i < _pts_tag_index[point].size() - 1; i++){
                                // if (_pts_tag_index[point].empty()) break;
                                uint32_t tag = _pts_tag_index[point][i].first;
                                new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(tag,tag_current_position));
                                if (tag == LARGE_TAG){
                                    float maximum_level = 0.0f;
                                    for (uint32_t subtag:_pts_to_labels[point]){
                                        if (hierarchy_level_copy[subtag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                            large_tags.push_back(subtag);
                                            maximum_level += hierarchy_level_copy[subtag];
                                        }
                                    }
                                    large_degree = (float)maximum_level/(float)block_level_sum*_graph_degree*_pack_k;   
                                    tag_current_position += large_degree;
                                }
                                else if (hierarchy_level_copy[tag]>=getLevel(CARDINALITY_THRESHOLD_LOW)){
                                    uint32_t cur_degree = (float)hierarchy_level_copy[tag]/(float)block_level_sum*_graph_degree*_pack_k;
                                    tag_current_position += cur_degree;
                                }
                            }
                            new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(-1,tag_current_position));
                            uint32_t end_position = block_start_point + tag_current_position;
                            new_graph_index[point] = std::pair<uint32_t,uint32_t>(block_start_point, end_position);
                            block_start_point = end_position;
                            uint32_t point_new_total_degree = new_graph_index[point].second - new_graph_index[point].first;
                            new_neighbors[point] = std::vector<uint32_t>(point_new_total_degree,std::numeric_limits<uint32_t>::max());
                            uint32_t* prev_neighbor_list = _graph_store + _pts_graph_index[point].first;
                            if (_pts_graph_index[point].first==std::numeric_limits<uint32_t>::max()){
                                std::cout<<"The start index has not been defined"<<std::endl;
                                exit(-1);
                            }
                            for (int i = 0; i < new_graph_degree[point].size()-1; i++){
                                // if (new_graph_degree[point].empty()) break;
                                uint32_t start_index = new_graph_degree[point][i].second;
                                uint32_t end_index = new_graph_degree[point][i+1].second;
                                uint32_t tag = new_graph_degree[point][i].first;
                                uint32_t degree = end_index - start_index;
                                uint32_t prev_start_index = _pts_tag_index[point][i].second;
                                uint32_t prev_end_index = _pts_tag_index[point][i+1].second;
                                uint32_t prev_degree = prev_end_index - prev_start_index;
                                if (tag!=_pts_tag_index[point][i].first){
                                    std::cout<<"Tag should be the same"<<std::endl;
                                    std::cout<<"Tag: "<<tag<<std::endl;
                                    std::cout<<"_pts_tag_index[point][i].first: "<<_pts_tag_index[point][i].first<<std::endl;
                                    std::cout<<"point: "<<point<<", i: "<<i<<std::endl;
                                    exit(-1);
                                }
                                if (degree == 0) continue;
                                if (tag == building_tag){
                                    _global_mtx.lock();
                                    std::vector<uint32_t> _label_to_pts_tag(_label_to_pts[tag]);
                                    _global_mtx.unlock();
                                    // brute force
                                    std::priority_queue<std::pair<float,uint32_t>> pq;
                                    for (int j = 0; j < _label_to_pts_tag.size(); j++){
                                        uint32_t candidate = _label_to_pts_tag[j];
                                        if (candidate == point) continue;
                                        float dist = _dist_fn->compare(_vec_store + point * _dim, _vec_store + candidate * _dim, _dim);
                                        pq.push(std::pair<float,uint32_t>(dist,candidate));
                                    }
                                    std::vector<uint32_t> neighbor_list;
                                    if (tag == LARGE_TAG){
                                        std::cerr<<"Tag shouldn't be LARGE_TAG"<<std::endl;
                                    }
                                    std::vector<uint32_t> target_tag(1,tag);
                                    pruning(point, pq, target_tag, neighbor_list, degree, alpha);
                                    memcpy(new_neighbors[point].data()+start_index,neighbor_list.data(),neighbor_list.size()*sizeof(uint32_t));
                                }
                                else {
                                    // copy from previous neighbors
                                    if (degree >= prev_degree && prev_degree > 0){
                                        memcpy(new_neighbors[point].data()+start_index, prev_neighbor_list+prev_start_index,prev_degree*sizeof(uint32_t));
                                    }
                                    if (degree < prev_degree){
                                        memcpy(new_neighbors[point].data()+start_index, prev_neighbor_list+prev_start_index,degree*sizeof(uint32_t));
                                    }
                                }
                            }
                            _label_lock[point_lock_id].unlock();
                        }
                        for (uint32_t point: block_pts){
                            uint32_t point_lock_id = getLockIndex(point);
                            _label_lock[point_lock_id].lock();
                            _pts_tag_index[point] = new_graph_degree[point];
                            _pts_graph_index[point] = new_graph_index[point];
                            memcpy(_graph_store + new_graph_index[point].first, new_neighbors[point].data(), new_neighbors[point].size()*sizeof(uint32_t));
                            _label_lock[point_lock_id].unlock();
                        }

                    }
                    
                    _global_mtx.lock();
                    _label_graph_check[building_tag] = true;
                    _global_mtx.unlock();
                }
            }
            
            
            for (uint32_t todo_id: todolist){
                std::vector<uint32_t> large_tags;
                for (int ii = 0; ii < _pts_tag_index[todo_id].size()-1; ii++){
                    uint32_t tag = _pts_tag_index[todo_id][ii].first;
                    if (std::find(labels_need_graph_build.begin(),labels_need_graph_build.end(),tag)!=labels_need_graph_build.end()) continue;
                    // search for candidate and then pruning
                    uint32_t degree = _pts_tag_index[todo_id][ii+1].second - _pts_tag_index[todo_id][ii].second;
                    
                    if (degree == 0) continue;
                    std::vector<uint32_t> neighbors;
                    neighbors.reserve(degree);
                    if (tag==LARGE_TAG){
                        for (uint32_t sub_tag: _pts_to_labels[todo_id]){
                            if (hierarchy_level_copy[sub_tag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                large_tags.push_back(sub_tag);
                            }
                        }
                        std::unordered_map<uint32_t, float> neighbors_map;
                        std::vector<uint32_t> candidates(Lsize,std::numeric_limits<uint32_t>::max());
                        std::vector<float> candidates_distance(Lsize,std::numeric_limits<float>::max());
                        for (uint32_t sub_tag: large_tags){
                            search(_vec_store+todo_id*_dim,std::vector<uint32_t>(1,sub_tag),Lsize,candidates.data(),candidates_distance.data(),ef_construction);
                            for (uint32_t ii = 0; ii < Lsize; ii++){
                                if (candidates[ii] < _nd){
                                    neighbors_map[candidates[ii]] = candidates_distance[ii];
                                }
                            }
                        }
                        std::priority_queue<std::pair<float, uint32_t>> pq;
                        for (auto p: neighbors_map){
                            pq.emplace(p.second, p.first);
                        }
                        pruning(todo_id, pq, large_tags, neighbors, degree, alpha);

                    }
                    else{
                        std::vector<uint32_t> candidates(Lsize,std::numeric_limits<uint32_t>::max());
                        std::vector<float> candidates_distance(Lsize,std::numeric_limits<float>::max());
                        search(_vec_store+todo_id*_dim,std::vector<uint32_t>(1,tag),Lsize,candidates.data(),candidates_distance.data(),ef_construction);
                        // pruning
                        if (candidates[0]>_nd) continue;
                        neighbors.emplace_back(candidates[0]);
                        uint32_t count = 1;
                        for (uint32_t i=1;i<Lsize;i++){
                            bool prune = false;
                            if (candidates[i]>_nd) break;
                            for (uint32_t neigh:neighbors){
                                float distance2 = _dist_fn->compare((_vec_store+neigh*_dim),(_vec_store+candidates[i]*_dim),_dim);
                                if (candidates_distance[i]>distance2*alpha){ // candidate e is closer to neighbor than to q
                                    prune=true;
                                    break;
                                }
                            }
                            if (!prune){
                                count++;
                                neighbors.push_back(candidates[i]);
                                if (count==degree) break;
                            }
                        }
                    }
                    

                    // set neighbors
                    uint32_t point_lock_id = getLockIndex(todo_id);
                    _label_lock[point_lock_id].lock();
                    uint32_t start_index = _pts_tag_index[todo_id][ii].second;
                    uint32_t* neighbor_link_list = _graph_store + _pts_graph_index[todo_id].first + start_index;
                    memcpy(neighbor_link_list, neighbors.data(), sizeof(uint32_t)*neighbors.size());
                    _label_lock[point_lock_id].unlock();

                    for (uint32_t neigh: neighbors){
                        uint32_t neigh_lock_id = getLockIndex(neigh);
                        _label_lock[neigh_lock_id].lock();
                        

                        uint32_t index_start_point = _pts_graph_index[neigh].first;
                        if (index_start_point==std::numeric_limits<uint32_t>::max()){
                            _label_lock[neigh_lock_id].unlock();
                            continue;
                        }
                        uint32_t index_1 = 0, index_2 = 0;
                        uint32_t neigh_block_id = index_start_point/(_pack_k * _graph_degree);
                        uint32_t neigh_block_lock_id = getLockIndex(neigh_block_id);
                        for (int j = 0; j < _pts_tag_index[neigh].size()-1; j++){
                            if (_pts_tag_index[neigh][j].first == tag){
                                index_1 = _pts_tag_index[neigh][j].second;
                                index_2 = _pts_tag_index[neigh][j+1].second;
                                break;
                            }
                        }
                        if (index_1==index_2) {
                            _label_lock[neigh_lock_id].unlock();
                            continue;
                        }
                        {   
                            uint32_t* start_pointer = _graph_store+index_start_point;
                            uint32_t* end_pointer = start_pointer+index_2-1;
                            start_pointer+=index_1;
                            uint32_t allocate_degree = index_2 - index_1;
                            if (*end_pointer<_nd){
                                end_pointer++;
                                std::priority_queue<std::pair<float, uint32_t>> neighbor_pq;
                                float neighbor_to_dis_neigh = _dist_fn->compare(_vec_store+todo_id*_dim, _vec_store+neigh*_dim,_dim);
                                neighbor_pq.emplace(std::pair<float,uint32_t>(neighbor_to_dis_neigh,todo_id));
                                for (uint32_t *neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                                    uint32_t neighbor_id = *neighbor_pointer;
                                    if (neighbor_id>_nd) break;
                                    float distance_neigh_nn = _dist_fn->compare(_vec_store+neigh*_dim,_vec_store+neighbor_id*_dim,_dim);
                                    neighbor_pq.emplace(std::pair<float,uint32_t>(distance_neigh_nn,neighbor_id));
                                }
                                std::vector<uint32_t> prune_result;
                                std::vector<uint32_t> target_tags;
                                if (tag != LARGE_TAG){
                                    target_tags.push_back(tag);
                                }
                                else{
                                    for (uint32_t subtag: _pts_to_labels[neigh]){
                                        if (hierarchy_level_copy[subtag]>=getLevel(CARDINALITY_THRESHOLD_HIGH)){
                                            target_tags.push_back(subtag);
                                        }
                                    }
                                }
                                pruning(neigh, neighbor_pq, target_tags, prune_result, allocate_degree, alpha);
                                memcpy(start_pointer,prune_result.data(),sizeof(uint32_t)*prune_result.size());
                            }
                            else {
                                end_pointer++;
                                for (; start_pointer < end_pointer; ++start_pointer){
                                    if (*start_pointer>=_nd) break;
                                }
                                *start_pointer = todo_id;
                            }
                        }
                        _label_lock[neigh_lock_id].unlock();
                    }
                }
            }


            _global_mtx.lock();
            for (uint32_t todo_id: todolist){
                for (uint32_t label:_pts_to_labels[todo_id]){
                    if (_label_to_medoid_id.find(label)==_label_to_medoid_id.end()){
                        _label_to_medoid_id[label] = std::vector<uint32_t>(1,todo_id);
                    }
                    else{
                        if (_label_to_medoid_id[label].size()<_num_of_entry_point){
                            _label_to_medoid_id[label].push_back(todo_id);
                        }
                        else{
                            uint32_t cur_num = _label_to_pts[label].size();
                            uint32_t rand_ind = rand() % cur_num;
                            if (rand_ind < _num_of_entry_point){
                                _label_to_medoid_id[label][rand_ind] = todo_id;
                            }
                        }
                    }
                }
            }
            _global_mtx.unlock();
        }

        void save(std::string& save_path_prefix){
            {
                std::string bin_file = save_path_prefix+"_storage.bin";
                std::ofstream writer(bin_file,std::ios::out);
                writer.write((char*)&_nd,sizeof(size_t));
                writer.write((char*)&_dim,sizeof(size_t));
                writer.write((char*)&_graph_degree,sizeof(size_t));
                writer.write((char*)&_r,sizeof(float));
                writer.write((char*)&CARDINALITY_THRESHOLD_HIGH,sizeof(float));
                writer.write((char*)_vec_store,_nd*_dim*sizeof(T));
                writer.write((char*)_graph_store,_nd*_graph_degree*sizeof(uint32_t));
                writer.close();
            }

            {
                std::string row_index_file = save_path_prefix+"_graph_row_index.bin";
                std::ofstream writer(row_index_file);
                writer.write((char*)&_nd,sizeof(size_t));
                for (uint32_t i=0;i<_nd;i++){
                    std::vector<std::pair<uint32_t, uint32_t>>& row_index = _pts_tag_index[i];
                    size_t length = row_index.size();
                    writer.write((char*)&length,sizeof(size_t));
                    uint32_t start_index = _pts_graph_index[i].first;
                    uint32_t end_index = _pts_graph_index[i].second;
                    writer.write((char*)&start_index,sizeof(uint32_t));
                    writer.write((char*)&end_index,sizeof(uint32_t));
                    for (uint32_t j=0;j<length;j++){
                        auto& p = row_index[j];
                        uint32_t first = p.first;
                        uint32_t second = p.second;
                        writer.write((char*)&first,sizeof(uint32_t));
                        writer.write((char*)&second,sizeof(uint32_t));
                    }
                }
                writer.close();
            }

            {
                std::string label_file = save_path_prefix+"_labels.bin";
                std::ofstream writer(label_file);
                writer.write((char*)&_nd,sizeof(size_t));
                for (uint32_t i=0;i<_nd;i++){
                    std::vector<uint32_t>& point_label = _pts_to_labels[i];
                    size_t label_num = point_label.size();
                    writer.write((char*)&label_num,sizeof(size_t));
                    writer.write((char*)point_label.data(),sizeof(uint32_t)*label_num);
                }
                writer.close();
            }

            {
                std::string label_file = save_path_prefix+"_label_to_pts.bin";
                std::ofstream writer(label_file);
                size_t label_num = _label_to_pts.size();
                std::cout<<"saving label_num: "<<label_num<<std::endl;
                writer.write((char*)&label_num,sizeof(size_t));
                for (auto p:_label_to_pts){
                    uint32_t label = p.first;
                    std::vector<uint32_t>& points = p.second;
                    writer.write((char*)&label,sizeof(uint32_t));
                    size_t point_num = points.size();
                    writer.write((char*)&point_num,sizeof(size_t));
                    writer.write((char*)points.data(),sizeof(uint32_t)*point_num);
                }
                writer.close();
            }

            {
                std::string hierarchy_file = save_path_prefix+"_hierarchy.bin";
                std::ofstream writer(hierarchy_file);
                size_t label_num = _label_hierarchy_level.size();
                writer.write((char*)&label_num,sizeof(size_t));
                for (auto p:_label_hierarchy_level){
                    uint32_t first = p.first;
                    float second = p.second;
                    writer.write((char*)&first,sizeof(uint32_t));
                    writer.write((char*)&second,sizeof(float));
                }
                writer.close();
            }

            {
                std::string graph_check = save_path_prefix+"_graph_check.bin";
                std::ofstream writer(graph_check);
                size_t label_num = _label_graph_check.size();
                writer.write((char*)&label_num,sizeof(size_t));
                for (auto p:_label_graph_check){
                    uint32_t first = p.first;
                    bool second = p.second;
                    writer.write((char*)&first,sizeof(uint32_t));
                    writer.write((char*)&second,sizeof(bool));
                }
                writer.close();
            }

            {
                std::string entry_point_file = save_path_prefix+"_entry_points.bin";
                std::ofstream writer(entry_point_file);
                size_t num_of_label = _label_to_medoid_id.size();
                writer.write((char*)&num_of_label,sizeof(size_t));
                for (auto& pair:_label_to_medoid_id){
                    uint32_t label = pair.first;
                    std::vector<uint32_t>& medoids = pair.second;
                    size_t medoid_num = medoids.size();
                    writer.write((char*)&label,sizeof(uint32_t));
                    writer.write((char*)&medoid_num,sizeof(size_t));
                    writer.write((char*)medoids.data(),sizeof(uint32_t)*medoid_num);
                }
                writer.close();
            }
  
            
        }

        void load(std::string& save_path_prefix){
            {
                std::string bin_file = save_path_prefix+"_storage.bin";
                std::ifstream reader(bin_file);
                reader.read((char*)&_nd,sizeof(size_t));
                reader.read((char*)&_dim,sizeof(size_t));
                reader.read((char*)&_graph_degree,sizeof(size_t));
                reader.read((char*)&_r,sizeof(float));
                reader.read((char*)&CARDINALITY_THRESHOLD_HIGH,sizeof(float));
                if (_vec_store==nullptr){
                    _vec_store = (T*)malloc(_nd*_dim*sizeof(T));
                }
                if (_graph_store==nullptr){
                    _graph_store = (uint32_t*)malloc(_nd*_graph_degree*sizeof(uint32_t));
                    memset(_graph_store, -1, sizeof(uint32_t)*_nd*_graph_degree);
                }
                reader.read((char*)_vec_store,_nd*_dim*sizeof(T));
                reader.read((char*)_graph_store,_nd*_graph_degree*sizeof(uint32_t));
                reader.close();
            }

            {
                std::string row_index_file = save_path_prefix+"_graph_row_index.bin";
                std::ifstream reader(row_index_file);
                reader.read((char*)&_nd,sizeof(size_t));
                _pts_tag_index.resize(_nd);
                _pts_graph_index.resize(_nd);
                for (uint32_t i=0;i<_nd;i++){
                    size_t length = 0;
                    reader.read((char*)&length,sizeof(size_t));
                    _pts_tag_index[i].resize(length);
                    uint32_t i1 = 0;
                    uint32_t i2 = 0;
                    reader.read((char*)&i1,sizeof(uint32_t));
                    reader.read((char*)&i2,sizeof(uint32_t));
                    _pts_graph_index[i] = std::pair<uint32_t,uint32_t>(i1,i2);
                    for (size_t j=0;j<length;j++){
                        uint32_t first = 0,second = 0;
                        reader.read((char*)&first,sizeof(uint32_t));
                        reader.read((char*)&second,sizeof(uint32_t));
                        _pts_tag_index[i][j] = std::pair<uint32_t,uint32_t>(first,second);
                    }
                }
                reader.close();
            }

            {
                std::string label_file = save_path_prefix+"_labels.bin";
                std::ifstream reader(label_file);
                reader.read((char*)&_nd,sizeof(size_t));
                _pts_to_labels.resize(_nd);
                for (uint32_t i=0;i<_nd;i++){
                    size_t label_num = 0;
                    reader.read((char*)&label_num,sizeof(size_t));
                    _pts_to_labels[i].resize(label_num);
                    reader.read((char*)_pts_to_labels[i].data(),sizeof(uint32_t)*label_num);
                }
                reader.close();
            }

            {
                std::string label_file = save_path_prefix+"_label_to_pts.bin";
                std::ifstream reader(label_file);
                size_t label_num, point_num;
                uint32_t label;
                reader.read((char*)&label_num,sizeof(size_t));
                std::cout<<"label_num: "<<label_num<<std::endl;
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&label,sizeof(uint32_t));
                    reader.read((char*)&point_num,sizeof(size_t));
                    std::vector<uint32_t> points(point_num,0);
                    reader.read((char*)points.data(),sizeof(uint32_t)*point_num);
                    _label_to_pts[label] = points;
                }
                reader.close();
                std::cout<<"_label_to_pts[0].size(): "<<_label_to_pts[0].size()<<std::endl;
            }

            {
                std::string hierarchy_file = save_path_prefix+"_hierarchy.bin";
                std::ifstream reader(hierarchy_file);
                size_t label_num;
                uint32_t first;
                float second;
                reader.read((char*)&label_num,sizeof(size_t));
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&first,sizeof(uint32_t));
                    reader.read((char*)&second,sizeof(float));
                    _label_hierarchy_level[first] = second;
                }
                reader.close();
            }

            {
                std::string graph_check_file = save_path_prefix+"_graph_check.bin";
                std::ifstream reader(graph_check_file);
                size_t label_num;
                uint32_t first;
                bool second;
                reader.read((char*)&label_num,sizeof(size_t));
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&first,sizeof(uint32_t));
                    reader.read((char*)&second,sizeof(bool));
                    _label_graph_check[first] = second;
                }
                reader.close();
            }

            {
                std::string entry_point_file = save_path_prefix+"_entry_points.bin";
                std::ifstream reader(entry_point_file);
                size_t label_num,num_of_ep;
                uint32_t label;
                _num_of_entry_point = 0;
                reader.read((char*)&label_num,sizeof(size_t));
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&label,sizeof(uint32_t));
                    reader.read((char*)&num_of_ep,sizeof(size_t));
                    std::vector<uint32_t> eps(num_of_ep,0);
                    reader.read((char*)eps.data(),sizeof(uint32_t)*num_of_ep);
                    _label_to_medoid_id[label] = eps;
                    if (_num_of_entry_point<num_of_ep){
                        _num_of_entry_point = num_of_ep;
                    }
                }
                reader.close();
            }
        }
};