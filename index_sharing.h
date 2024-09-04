#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <mutex>
#include <queue>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "distance.h"
#include "io.h"

#define MAX_LOCKS 65536
#define getLockIndex(a) a&65535


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

        FilterIndex(size_t nd, size_t dim, size_t graph_degree, Distance<T>* dist_fn, uint32_t num_of_entry_point = 5, uint32_t pack_k = 10){
            _dim = dim;
            _nd = nd;
            _graph_degree = graph_degree;
            _vec_store = (T*)malloc(sizeof(T)*nd*dim);

            _graph_store = (uint32_t*)malloc(sizeof(uint32_t)*nd*graph_degree);
            memset(_graph_store, -1, sizeof(uint32_t)*_nd*_graph_degree);
            _pts_to_labels.resize(nd);
            _pts_tag_index.resize(nd);
            _pts_graph_index = std::vector<std::pair<uint32_t,uint32_t>>(nd, std::pair<uint32_t,uint32_t>(std::numeric_limits<uint32_t>::max(),-1));
            _label_lock = std::vector<std::mutex>(MAX_LOCKS);
            _dist_fn = dist_fn;
            _num_of_entry_point = num_of_entry_point;
            _pack_k = pack_k;
            curBlockID = 0;
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

        uint32_t search(T* query, std::vector<uint32_t> labels, uint32_t topk, uint32_t* ids_res, float* dis_res, uint32_t ef_search, bool verbose=false){
            // cardinality estimation
            uint32_t cardinality = _nd;
            uint32_t smallest_cardinality = _nd;
            uint32_t smallest_filter = labels[0];
            uint32_t computation_count = 0;
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
                for (uint32_t i=0;i<label_size.size();i++){
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
            // bool use_graph = (cardinality >= 1000 || smallest_cardinality >= 4096) && smallest_graph_search;
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
                for (uint32_t i=0;i<candidates.size();i++){
                    if (i<candidates.size()-1){
                        _mm_prefetch(_vec_store+candidates[i+1]*_dim,_MM_HINT_T0);
                    }
                    float dis = _dist_fn->compare(_vec_store+candidates[i]*_dim,query,_dim);
                    computation_count++;
                    pq.emplace(dis,candidates[i]);
                    if (pq.size()>topk){
                        pq.pop();
                    }
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

                    uint32_t start_index = 0, end_index = 0;
                    for (uint32_t i=0;i<_pts_tag_index[id].size();i++){
                        std::pair<uint32_t,uint32_t>& row_index = _pts_tag_index[id][i];
                        if (row_index.first == smallest_filter){
                            start_index = row_index.second;
                            end_index = _pts_tag_index[id][i+1].second;
                            break;
                        }
                    }
                    uint32_t candidate_num = end_index - start_index;
                    uint32_t* start_pointer = _graph_store + _pts_graph_index[id].first;
                    uint32_t* end_pointer = start_pointer+end_index;
                    start_pointer+=start_index;
                    candidates.clear();
                    for (uint32_t* neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                        uint32_t neighbor_id=*neighbor_pointer;
                        if (neighbor_id>_nd) break;
                        if (std::find(visit_set.begin(),visit_set.end(),neighbor_id)==visit_set.end()){
                            visit_set.push_back(neighbor_id);
                            candidates.push_back(neighbor_id);
                        }
                    }

                    for (uint32_t i=0;i<candidates.size();i++){
                        if (i<candidates.size()-1){
                            _mm_prefetch(_vec_store+candidates[i+1]*_dim,_MM_HINT_T0);
                        }
                        float candidate_dis = _dist_fn->compare(_vec_store+candidates[i]*_dim,query,_dim);
                        computation_count++;
                        if (top_candidates.size()<ef_search||lower_bound>candidate_dis){
                            ef_queue.emplace(-candidate_dis,candidates[i]);
                            top_candidates.emplace(candidate_dis,candidates[i]);
			    bool match_filter = match_all_filters(candidates[i],labels);
                            if (match_filter){
                            	pq.emplace(candidate_dis,candidates[i]);
                            }
                            if (top_candidates.size()>ef_search){
                                top_candidates.pop();
                            	if (!top_candidates.empty()){
                                	lower_bound=top_candidates.top().first;
                            	}
			    }
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


        void addPoint(T* data, size_t id, std::vector<uint32_t>& tags, uint32_t Lsize = 100, uint32_t ef_construction=100, float alpha = 1.2f){
            if (tags.empty()){
                tags.emplace_back(std::numeric_limits<uint32_t>::max());
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
            _global_mtx.lock();
            for (uint32_t todo_id: todolist){
                for (uint32_t tag: _pts_to_labels[todo_id]){
                    float cur_level_for_label = std::max(1.0, std::log(_label_to_pts[tag].size())/std::log(32));
                    if (_label_hierarchy_level[tag] < 2 && cur_level_for_label>=2){
                        labels_need_graph_build.emplace_back(tag);
                    }
                    if (cur_level_for_label>=2){
                        level_sum += cur_level_for_label;
                    }
                    _label_hierarchy_level[tag] = cur_level_for_label;
                }
            }
            uint32_t current_position = allocate_start_point;
            for (uint32_t todo_id: todolist){
                uint32_t start_position = current_position;
                uint32_t tag_current_position = 0;
                for (uint32_t tag:_pts_to_labels[todo_id]){
                    _pts_tag_index[todo_id].push_back(std::pair<uint32_t,uint32_t>(tag,tag_current_position));
                    if (_label_hierarchy_level[tag]>=2){
                        uint32_t degree = (float)_label_hierarchy_level[tag]/(float)level_sum*_graph_degree*_pack_k;
                        tag_current_position += degree;
                    }
                }
                _pts_tag_index[todo_id].push_back(std::pair<uint32_t,uint32_t>(-1,tag_current_position));
                current_position = start_position + tag_current_position;
                _pts_graph_index[todo_id] = std::pair<uint32_t,uint32_t>(start_position,current_position);
            }
            _global_mtx.unlock();
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
                        for (uint32_t point: _block_to_pts[block]){
                            new_graph_degree[point] = std::vector<std::pair<uint32_t, uint32_t>>();
                            for (uint32_t tag: _pts_to_labels[point]){
                                if (_label_hierarchy_level[tag]>=2){
                                    block_level_sum += _label_hierarchy_level[tag];
                                }
                            }
                        }
                        uint32_t block_start_point = block * _pack_k * _graph_degree;
                        for (uint32_t point: _block_to_pts[block]){
                            uint32_t tag_current_position = 0;
                            for (uint32_t tag: _pts_to_labels[point]){
                                new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(tag,tag_current_position));
                                if (_label_hierarchy_level[tag]>=2){
                                    uint32_t cur_degree = (float)_label_hierarchy_level[tag]/(float)block_level_sum*_graph_degree*_pack_k;
                                    tag_current_position += cur_degree;
                                }
                            }
                            new_graph_degree[point].push_back(std::pair<uint32_t,uint32_t>(-1,tag_current_position));
                            uint32_t end_position = block_start_point + tag_current_position;
                            new_graph_index[point] = std::pair<uint32_t,uint32_t>(block_start_point, end_position);
                            block_start_point = end_position;
                        }
                        std::vector<uint32_t> block_pts(_block_to_pts[block]);
                        _global_mtx.unlock();
                        uint32_t block_lock_id = getLockIndex(block);
                        _label_lock[block_lock_id].lock();
                        for (uint32_t point: block_pts){
                            uint32_t point_new_total_degree = new_graph_index[point].second - new_graph_index[point].first;
                            new_neighbors[point] = std::vector<uint32_t>(point_new_total_degree,std::numeric_limits<uint32_t>::max());
                            uint32_t* prev_neighbor_list = _graph_store + _pts_graph_index[point].first;
                            if (_pts_graph_index[point].first==std::numeric_limits<uint32_t>::max()){
                                std::cout<<"The start index has not been defined"<<std::endl;
                                exit(-1);
                            }
                            for (uint32_t i = 0; i < new_graph_degree[point].size()-1; i++){
                                uint32_t start_index = new_graph_degree[point][i].second;
                                uint32_t end_index = new_graph_degree[point][i+1].second;
                                uint32_t tag = new_graph_degree[point][i].first;
                                uint32_t degree = end_index - start_index;
                                uint32_t prev_start_index = _pts_tag_index[point][i].second;
                                uint32_t prev_end_index = _pts_tag_index[point][i+1].second;
                                uint32_t prev_degree = prev_end_index - prev_start_index;
                                if (tag!=_pts_tag_index[point][i].first){
                                    std::cout<<"Tag should be the same"<<std::endl;
                                    exit(-1);
                                }
                                if (degree == 0) continue;
                                if (tag == building_tag){
                                    _global_mtx.lock();
                                    std::vector<uint32_t> _label_to_pts_tag(_label_to_pts[tag]);
                                    _global_mtx.unlock();
                                    // brute force
                                    std::priority_queue<std::pair<float,uint32_t>> pq;
                                    for (uint32_t j = 0; j < _label_to_pts_tag.size(); j++){
                                        uint32_t candidate = _label_to_pts_tag[j];
                                        if (candidate == point) continue;
                                        float dist = _dist_fn->compare(_vec_store + point * _dim, _vec_store + candidate * _dim, _dim);
                                        pq.push(std::pair<float,uint32_t>(-dist,candidate));
                                    }
                                    std::vector<uint32_t> neighbor_list;
                                    while (!pq.empty()){
                                        auto p = pq.top();
                                        pq.pop();
                                        bool good = true;
                                        uint32_t candidate = p.second;
                                        float dis_target_cand = -p.first;
                                        for (uint32_t existing_neigh: neighbor_list){
                                            float dis_cand_neigh = _dist_fn->compare(_vec_store + existing_neigh * _dim, _vec_store + candidate * _dim, _dim);
                                            if (dis_cand_neigh * alpha < dis_target_cand){
                                                good = false;
                                                break;
                                            }
                                        }
                                        if (good){
                                            neighbor_list.push_back(candidate);
                                            if (neighbor_list.size()>=degree) break;
                                        }
                                    }
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
                        }
                        
                        // update _pts_graph_index, _pts_tag_index and _graph_store
                        for (uint32_t point: block_pts){
                            _pts_tag_index[point] = new_graph_degree[point];
                            _pts_graph_index[point] = new_graph_index[point];
                            memcpy(_graph_store + new_graph_index[point].first, new_neighbors[point].data(), new_neighbors[point].size()*sizeof(uint32_t));
                        }
                        _label_lock[block_lock_id].unlock();
                    }
                    _global_mtx.lock();
                    _label_graph_check[building_tag] = true;
                    _global_mtx.unlock();
                }
            }
            uint32_t todoblock_lock_id = getLockIndex(current_block_id);
            for (uint32_t todo_id: todolist){
                for (uint32_t ii = 0; ii < _pts_tag_index[todo_id].size()-1; ii++){
                    uint32_t tag = _pts_tag_index[todo_id][ii].first;
                    if (std::find(labels_need_graph_build.begin(),labels_need_graph_build.end(),tag)!=labels_need_graph_build.end()) continue;
                    // search for candidate and then pruning
                    uint32_t degree = _pts_tag_index[todo_id][ii+1].second - _pts_tag_index[todo_id][ii].second;
                    
                    if (degree == 0) continue;
                    std::vector<uint32_t> candidates(Lsize,std::numeric_limits<uint32_t>::max());
                    std::vector<float> candidates_distance(Lsize,std::numeric_limits<float>::max());
                    search(_vec_store+todo_id*_dim,std::vector<uint32_t>(1,tag),Lsize,candidates.data(),candidates_distance.data(),ef_construction);
                    // pruning
                    std::vector<uint32_t> neighbors;
                    neighbors.reserve(degree);
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

                    // set neighbors
                    _label_lock[todoblock_lock_id].lock();
                    uint32_t start_index = _pts_tag_index[todo_id][ii].second;
                    uint32_t* neighbor_link_list = _graph_store + _pts_graph_index[todo_id].first + start_index;
                    memcpy(neighbor_link_list, neighbors.data(), sizeof(uint32_t)*neighbors.size());
                    _label_lock[todoblock_lock_id].unlock();

                    for (uint32_t neigh: neighbors){
                        uint32_t index_start_point = _pts_graph_index[neigh].first;
                        if (index_start_point==std::numeric_limits<uint32_t>::max()) continue;
                        uint32_t index_1 = 0, index_2 = 0;
                        uint32_t neigh_block_id = index_start_point/(_pack_k * _graph_degree);
                        uint32_t neigh_block_lock_id = getLockIndex(neigh_block_id);
                        _label_lock[neigh_block_lock_id].lock();
                        for (uint32_t j = 0; j < _pts_tag_index[neigh].size()-1; j++){
                            if (_pts_tag_index[neigh][j].first == tag){
                                index_1 = _pts_tag_index[neigh][j].second;
                                index_2 = _pts_tag_index[neigh][j+1].second;
                                break;
                            }
                        }
                        if (index_1==index_2) {
                            _label_lock[neigh_block_lock_id].unlock();
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
                                neighbor_pq.emplace(std::pair<float,uint32_t>(-neighbor_to_dis_neigh,todo_id));
                                for (uint32_t *neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                                    uint32_t neighbor_id = *neighbor_pointer;
                                    if (neighbor_id>_nd) break;
                                    float distance_neigh_nn = _dist_fn->compare(_vec_store+neigh*_dim,_vec_store+neighbor_id*_dim,_dim);
                                    neighbor_pq.emplace(std::pair<float,uint32_t>(-distance_neigh_nn,neighbor_id));
                                }
                                std::vector<uint32_t> prune_result;
                                prune_result.reserve(allocate_degree);
                                while (!neighbor_pq.empty()){
                                    auto neighbor_pair = neighbor_pq.top();
                                    neighbor_pq.pop();
                                    float dis1 = -neighbor_pair.first;
                                    uint32_t neighbor_id = neighbor_pair.second;
                                    bool prune = false;
                                    for (uint32_t second_pair: prune_result){
                                        float dis2 = _dist_fn->compare(_vec_store+second_pair*_dim,_vec_store+neighbor_id*_dim,_dim);
                                        if (alpha*dis2 < dis1){
                                            prune = true;
                                            break;
                                        }
                                    }
                                    if (!prune){
                                        prune_result.emplace_back(neighbor_id);
                                        if (prune_result.size()>=allocate_degree) break;
                                    }
                                }
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
                        _label_lock[neigh_block_lock_id].unlock();
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


