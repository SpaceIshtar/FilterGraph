#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <mutex>
#include <queue>
#include <cmath>
#include <algorithm>

#include "distance.h"
#include "io.h"

#define MAX_LOCKS 65536
#define getLockIndex(a) a&65535


template<typename T>
class FilterIndex{
    public:
        
        char* _storage = nullptr; // each point will save vector + neighbors
        size_t _dim = 0;
        size_t _nd = 0;
        size_t _graph_degree = 0;
        Distance<T>* _dist_fn = nullptr;
        size_t _graph_offset = 0; // set when initialization, equals = _nd * sizeof(T)
        size_t _point_size = 0; // dim*sizeof(T)+degree*sizeof(uint32_t)
        uint32_t _num_of_entry_point = 5;
        uint32_t sample_size = 10000;
        

        std::vector<std::vector<uint32_t>> _pts_to_labels;
        std::unordered_map<uint32_t,std::vector<uint32_t>> _label_to_pts;
        std::unordered_map<uint32_t,std::vector<uint32_t>> _label_to_medoid_id;
        std::unordered_map<uint32_t,std::vector<uint32_t>> _label_to_sample_id;
        std::unordered_map<uint32_t, bool> _label_graph_check;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> _graph_row_index;
        std::unordered_map<uint32_t, uint32_t> _label_hierarchy_level;
        mutable std::vector<std::mutex> _label_lock;
        std::mutex _global_mtx;

        FilterIndex(Distance<T>* dist_fn){
            _dist_fn = dist_fn;
            _label_lock = std::vector<std::mutex>(MAX_LOCKS);
        }

        FilterIndex(size_t nd, size_t dim, size_t graph_degree, Distance<T>* dist_fn, uint32_t num_of_entry_point = 5){
            _dim = dim;
            _nd = nd;
            _graph_degree = graph_degree;
            _graph_offset = _dim*sizeof(T);
            _point_size = _graph_offset + graph_degree*sizeof(uint32_t);
            _storage = (char*)malloc(_nd*_point_size);
            _pts_to_labels.resize(nd);
            _graph_row_index.resize(nd);
            _label_lock = std::vector<std::mutex>(MAX_LOCKS);
            _dist_fn = dist_fn;
            _num_of_entry_point = num_of_entry_point;
        }

        ~FilterIndex(){
            free(_storage);
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
            std::sort(labels.begin(),labels.end());
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
            bool use_graph = (cardinality >= 1024 || smallest_cardinality >= 4096) && smallest_graph_search;
            if (!use_graph){
        //    if (cardinality < 1000 || smallest_cardinality < 1000 || !smallest_graph_search){
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
                        _mm_prefetch(_storage+candidates[i+1]*_point_size,_MM_HINT_T0);
                    }
                    float dis = _dist_fn->compare((T*)(_storage+candidates[i]*_point_size),query,_dim);
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
                    float dis = _dist_fn->compare((T*)(_storage+entry_id*_point_size),query,_dim);
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
                    for (uint32_t i=0;i<_graph_row_index[id].size();i++){
                        std::pair<uint32_t,uint32_t>& row_index = _graph_row_index[id][i];
                        if (row_index.first == smallest_filter){
                            start_index = row_index.second;
                            end_index = _graph_row_index[id][i+1].second;
                            break;
                        }
                    }
                    uint32_t candidate_num = end_index - start_index;

                    uint32_t* start_pointer = (uint32_t*)(_storage+id*_point_size+_graph_offset);
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
                            _mm_prefetch(_storage+candidates[i+1]*_point_size,_MM_HINT_T0);
                        }
                        float candidate_dis = _dist_fn->compare((T*)(_storage+candidates[i]*_point_size),query,_dim);
                        computation_count++;
                        if (top_candidates.size()<ef_search||lower_bound>candidate_dis){
                            ef_queue.emplace(-candidate_dis,candidates[i]);
                            top_candidates.emplace(candidate_dis,candidates[i]);
                            if (match_all_filters(candidates[i],labels)){
                                pq.emplace(candidate_dis,candidates[i]);
                            }
                            if (top_candidates.size()>ef_search){
                                    top_candidates.pop();
                                }
                                if (!top_candidates.empty()){
                                    lower_bound=top_candidates.top().first;
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

        void addPoint(T* data, size_t id, std::vector<uint32_t>& labels, uint32_t Lsize = 100, uint32_t ef_construction=100){
            if (labels.empty()){
                labels.emplace_back(std::numeric_limits<uint32_t>::max());
            }

            {
                uint32_t lock_id = getLockIndex(id);
                _label_lock[lock_id].lock();
                memcpy(_storage+id*_point_size,data,_graph_offset);
                memset(_storage+id*_point_size+_graph_offset,std::numeric_limits<uint32_t>::max(),_graph_degree*sizeof(uint32_t));
                _label_lock[lock_id].unlock();
                _global_mtx.lock();
                _pts_to_labels[id] = labels;
                for (uint32_t label: labels){
                    _label_graph_check[label] = true;
                }
                _global_mtx.unlock();
                std::sort(_pts_to_labels[id].begin(),_pts_to_labels[id].end());
            }

            _global_mtx.lock();
            for (uint32_t label:labels){
                _label_hierarchy_level[label] = std::max(1.0,std::log2(_label_to_pts[label].size()));
            }

            // allocate neighborhood
            {
                uint32_t level_num = 0;
                for (uint32_t label:labels){
                    level_num+=_label_hierarchy_level[label];
                }
                uint32_t current_position = 0;
                for (uint32_t label:labels){
                    _graph_row_index[id].push_back(std::pair<uint32_t,uint32_t>(label,current_position));
                    current_position+=(float)_label_hierarchy_level[label]/(float)level_num*_graph_degree;
                }
                _graph_row_index[id].push_back(std::pair<uint32_t,uint32_t>(-1,_graph_degree));
            }
            _global_mtx.unlock();

            for (uint32_t ii=1;ii<_graph_row_index[id].size();ii++){
                uint32_t label = _graph_row_index[id][ii-1].first;
                uint32_t degree = _graph_row_index[id][ii].second - _graph_row_index[id][ii-1].second;
                if (degree == 0) continue;
                std::vector<uint32_t> candidates(Lsize,std::numeric_limits<uint32_t>::max());
                std::vector<float> candidates_distance(Lsize,std::numeric_limits<float>::max());
                search(data,std::vector<uint32_t>(1,label),Lsize,candidates.data(),candidates_distance.data(),ef_construction);
                // pruning
                std::vector<uint32_t> neighbors;
                neighbors.reserve(degree);
                std::unordered_map<uint32_t,float> neighbor_to_dis;
                if (candidates[0]>_nd) continue;
                neighbors.emplace_back(candidates[0]);
                uint32_t count = 1;
                for (uint32_t i=1;i<Lsize;i++){
                    bool prune = false;
                    if (candidates[i]>_nd) break;
                    for (uint32_t neigh:neighbors){
                        float distance2 = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+candidates[i]*_point_size),_dim);
                        if (candidates_distance[i]>distance2){ // candidate e is closer to neighbor than to q
                            prune=true;
                            break;
                        }
                    }
                    if (!prune){
                        count++;
                        neighbors.push_back(candidates[i]);
                        neighbor_to_dis[candidates[i]]=candidates_distance[i];
                        if (count==degree) break;
                    }
                }
                
                // set neighbors
                uint32_t lock_id = getLockIndex(id);
                _label_lock[lock_id].lock();
                uint32_t start_index = _graph_row_index[id][ii-1].second;
                memcpy(_storage+id*_point_size+_graph_offset+start_index*sizeof(uint32_t),neighbors.data(),neighbors.size()*sizeof(uint32_t));
                _label_lock[lock_id].unlock();

                // insert id into neighbors' neighbor list
                for (uint32_t neigh:neighbors){
                    // get label neighbor
                    uint32_t index_1 = 0, index_2 = 0;
                    for (uint32_t i=0;i<_graph_row_index[neigh].size();i++){
                        auto& p = _graph_row_index[neigh][i];
                        if (p.first == label){
                            index_1 = p.second;
                            index_2 = _graph_row_index[neigh][i+1].second;
                            break;
                        }
                    }
                    {   
                        uint32_t neighbor_lock_id = getLockIndex(neigh);
                        _label_lock[neighbor_lock_id].lock();
                        uint32_t* start_pointer = (uint32_t*)(_storage+neigh*_point_size+_graph_offset);
                        uint32_t* end_pointer = start_pointer+index_2;
                        start_pointer+=index_1;
                        uint32_t allocate_degree = index_2 - index_1;
                        std::priority_queue<std::pair<float, uint32_t>> neighbor_pq;
                        neighbor_pq.emplace(std::pair<float,uint32_t>(-neighbor_to_dis[neigh],id));
                        for (uint32_t *neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                            uint32_t neighbor_id = *neighbor_pointer;
                            if (neighbor_id>_nd) break;
                            float distance_neigh_nn = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+neighbor_id*_point_size),_dim);
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
                                float dis2 = _dist_fn->compare((T*)(_storage+second_pair*_point_size),(T*)(_storage+neighbor_id*_point_size),_dim);
                                if (dis2 < dis1){
                                    prune = true;
                                    break;
                                }
                            }
                            if (!prune){
                                prune_result.emplace_back(neighbor_id);
                                if (prune_result.size()>=allocate_degree) break;
                            }
                        }
                        memset(start_pointer,std::numeric_limits<uint32_t>::max(),sizeof(uint32_t)*allocate_degree);
                        memcpy(start_pointer,prune_result.data(),sizeof(uint32_t)*prune_result.size());
                        _label_lock[neighbor_lock_id].unlock();
                    }
                }
            }

            _global_mtx.lock();
            for (uint32_t label:labels){
                if (_label_to_pts.find(label)==_label_to_pts.end()){
                    _label_to_pts[label] = std::vector<uint32_t>(1,id);
                    _label_to_medoid_id[label] = std::vector<uint32_t>(1,id);
                }
                else{
                    _label_to_pts[label].push_back(id);
                    if (_label_to_medoid_id[label].size()<_num_of_entry_point){
                        _label_to_medoid_id[label].push_back(id);
                    }
                    else{
                        uint32_t cur_num = _label_to_pts[label].size();
                        uint32_t rand_ind = rand() % cur_num;
                        if (rand_ind < _num_of_entry_point){
                            _label_to_medoid_id[label][rand_ind] = id;
                        }
                    }
                }
            }
            _global_mtx.unlock();
            
        }        

        void addPoint_v2(T* data, size_t id, std::vector<uint32_t>& labels, uint32_t Lsize = 100, uint32_t ef_construction = 100, float alpha = 1.2){
            if (labels.empty()){
                labels.emplace_back(std::numeric_limits<uint32_t>::max());
            }
            std::sort(labels.begin(),labels.end());
            {
                uint32_t lock_id = getLockIndex(id);
                _label_lock[lock_id].lock();
                memcpy(_storage+id*_point_size,data,_graph_offset);
                memset(_storage+id*_point_size+_graph_offset,std::numeric_limits<uint32_t>::max(),_graph_degree*sizeof(uint32_t));
                _label_lock[lock_id].unlock();
                _global_mtx.lock();
                _pts_to_labels[id] = labels;
                _global_mtx.unlock();
                std::sort(_pts_to_labels[id].begin(),_pts_to_labels[id].end());
            }

            std::vector<uint32_t> labels_need_graph_build;
            {
                uint32_t level_num = 0;
                _global_mtx.lock();
                for (uint32_t label:labels){
                    uint32_t cur_level_for_label = std::max(1.0,std::log2(_label_to_pts[label].size()));
                    if (_label_hierarchy_level[label] < 10 && cur_level_for_label>=10){
                        labels_need_graph_build.emplace_back(label);
                    }
                    _label_hierarchy_level[label] = cur_level_for_label;
                    if (_label_hierarchy_level[label]>=10){
                        level_num += _label_hierarchy_level[label];
                    }
                }
                uint32_t current_position = 0;
                for (uint32_t label:labels){
                    _graph_row_index[id].push_back(std::pair<uint32_t,uint32_t>(label,current_position));
                    if (_label_hierarchy_level[label]>=10){
                        uint32_t cur_degree = (float)_label_hierarchy_level[label]/(float)level_num*_graph_degree;
                        current_position+=cur_degree;
                    }
                }
                _graph_row_index[id].push_back(std::pair<uint32_t,uint32_t>(-1,current_position));
                _global_mtx.unlock();
            }            

            {

                for (uint32_t label:labels_need_graph_build){
                    // reallocate neighbor for all points
                    _global_mtx.lock();
                    std::vector<uint32_t> cur_label_to_points(_label_to_pts[label]);
                    std::vector<std::unordered_map<uint32_t,uint32_t>> cur_label_new_graph_degree;
                    uint32_t cur_label_point_num = cur_label_to_points.size();
                    for (uint32_t i = 0; i < cur_label_point_num; i++){
                        uint32_t cur_label_point_id = cur_label_to_points[i];
                        uint32_t level_num = 0;
                        std::unordered_map<uint32_t,uint32_t> cur_label_cur_point_graph_degree;
                        for (uint32_t cur_point_label: _pts_to_labels[cur_label_point_id]){
                            float cur_level_for_label = std::max(1.0,std::log2(_label_to_pts[cur_point_label].size()));
                            // float cur_level_for_label = std::max(1.0f,(float)(_label_to_pts[cur_point_label].size()));
                            if (cur_level_for_label>=10){
                                level_num+=cur_level_for_label;
                            }
                        }
                        for (uint32_t cur_point_label:_pts_to_labels[cur_label_point_id]){
                            if (_label_hierarchy_level[cur_point_label]>=10){
                                uint32_t cur_degree = (float)_label_hierarchy_level[cur_point_label]/(float)level_num*_graph_degree;
                                cur_label_cur_point_graph_degree.emplace(cur_point_label,cur_degree);
                            }
                            else{
                                cur_label_cur_point_graph_degree.emplace(cur_point_label,0);
                            }
                        }
                        cur_label_new_graph_degree.push_back(cur_label_cur_point_graph_degree);
                    }

                    _global_mtx.unlock();
                    // build graph for some labels
                    cur_label_point_num+=1;
                    cur_label_to_points.push_back(id);
                    std::vector<std::priority_queue<std::pair<float,uint32_t>>> pq_vector(cur_label_point_num,std::priority_queue<std::pair<float,uint32_t>>());
                    for (uint32_t i = 0; i < cur_label_point_num - 1; i++){
                        uint32_t i_id = cur_label_to_points[i];
                        for (uint32_t j = i+1; j < cur_label_point_num; j++){
                            uint32_t j_id = cur_label_to_points[j];
                            float distance = _dist_fn->compare((T*)(_storage+i_id*_point_size),(T*)(_storage+j_id*_point_size),_dim);
                            pq_vector[i].emplace(distance,j_id);
                            pq_vector[j].emplace(distance,i_id);
                        }
                    }
                    // set neighbors for previously inserted points
                    for (uint32_t i = 0; i < cur_label_point_num - 1; i++){
                        uint32_t i_id = cur_label_to_points[i];
                        uint32_t lock_id = getLockIndex(i_id);
                        _label_lock[lock_id].lock();
                        std::vector<std::pair<uint32_t, uint32_t>>& row_index = _graph_row_index[i_id];
                        std::vector<std::pair<uint32_t, uint32_t>> new_row_index;
                        std::vector<uint32_t> new_neighbors(_graph_degree,-1);
                        uint32_t current_position = 0;
                        uint32_t original_current_position = 0;
                        uint32_t label_pos = 0;
                        //set neighbors for other labels
                        for (uint32_t j = 0; j < row_index.size() - 1; j++){
                            uint32_t label_name = row_index[j].first;
                            if (label == label_name){
                                label_pos = current_position;
                            }
                            new_row_index.emplace_back(label_name,current_position);
                            uint32_t label_degree = row_index[j+1].second - row_index[j].second;
                            uint32_t new_label_degree = cur_label_new_graph_degree[i][label_name];
                            uint32_t start_index = original_current_position;
                            uint32_t* pointer = (uint32_t*)(_storage + i_id*_point_size+_graph_offset);
                            pointer+=start_index;
                            if (label_degree <= new_label_degree){
                                // copy from original neighbor
                                memcpy(new_neighbors.data()+current_position,pointer,sizeof(uint32_t)*label_degree);
                            }
                            else{
                                memcpy(new_neighbors.data()+current_position,pointer,sizeof(uint32_t)*new_label_degree);
                            }
                            current_position += new_label_degree;
                            original_current_position += label_degree;
                        }
                        new_row_index.emplace_back(-1,current_position);
                        //set neighbor for new label
                        uint32_t current_label_degree = cur_label_new_graph_degree[i][label];
                        //pruning
                        std::priority_queue<std::pair<float,uint32_t>> current_point_pq;
                        while (!pq_vector[i].empty()){
                            auto p = pq_vector[i].top();
                            pq_vector[i].pop();
                            current_point_pq.emplace(-p.first,p.second);
                        }
                        std::vector<uint32_t> candidates;
                        while (!current_point_pq.empty()){
                            float dis_cand_q = -current_point_pq.top().first;
                            uint32_t candidate_name = current_point_pq.top().second;
                            current_point_pq.pop();
                            bool pruning = false;
                            for (uint32_t neigh: candidates){
                                float dis_cand_neigh = _dist_fn->compare((T*)(_storage+candidate_name*_point_size),(T*)(_storage+neigh*_point_size),_dim);
                                if (alpha*dis_cand_neigh < dis_cand_q){
                                    pruning = true;
                                    break;
                                }
                            }
                            if (!pruning){
                                candidates.emplace_back(candidate_name);
                                if (candidates.size()>=current_label_degree) break;
                            }
                        }
                        memcpy(new_neighbors.data()+label_pos,candidates.data(),candidates.size()*sizeof(uint32_t));
                        
                        
                        _graph_row_index[i_id] = new_row_index;
                        memcpy(_storage+i_id*_point_size+_graph_offset,new_neighbors.data(),new_neighbors.size()*sizeof(uint32_t));
                        _label_lock[lock_id].unlock();
                    }
                    
                    // set neighbor for this newly inserted id point
                    uint32_t id_position = cur_label_point_num - 1;
                    uint32_t lock_id = getLockIndex(id);
                    _label_lock[lock_id].lock();
                    std::vector<std::pair<uint32_t, uint32_t>>& row_index = _graph_row_index[id];
                    std::vector<uint32_t> candidates;
                    for (uint32_t j = 0; j < row_index.size() - 1; j++){
                        uint32_t label_name = row_index[j].first;
                        uint32_t degree = row_index[j+1].second - row_index[j].second;
                        if (label_name != label){
                            continue;
                        }
                        candidates.clear();
                        uint32_t graph_index_current_postition = row_index[j].second;
                        uint32_t* pointer = (uint32_t*)(_storage + id*_point_size+_graph_offset);
                        pointer += graph_index_current_postition;
                        // pruning
                        std::priority_queue<std::pair<float,uint32_t>> current_point_pq;
                        uint32_t top_neighbor_internal_id = 1;
                        while (!pq_vector[id_position].empty()){
                            auto p = pq_vector[id_position].top();
                            pq_vector[id_position].pop();
                            current_point_pq.emplace(-p.first,p.second);
                        }
                        
                        std::vector<uint32_t> new_neighbors(degree,-1);
                        while (!current_point_pq.empty()){
                            float dis_cand_q = -current_point_pq.top().first;
                            uint32_t candidate_name = current_point_pq.top().second;
                            current_point_pq.pop();
                            bool pruning = false;
                            for (uint32_t neigh: candidates){
                                float dis_cand_neigh = _dist_fn->compare((T*)(_storage+candidate_name*_point_size),(T*)(_storage+neigh*_point_size),_dim);
                                if (alpha*dis_cand_neigh < dis_cand_q){
                                    pruning = true;
                                    break;
                                }
                            }
                            if (!pruning){
                                candidates.emplace_back(candidate_name);
                                if (candidates.size()>=degree) {
                                    break;
                                }
                            }
                        }
                        memcpy(pointer,candidates.data(),candidates.size()*sizeof(uint32_t));
                    }
                    
                    _label_lock[lock_id].unlock();
                    

                    _global_mtx.lock();
                    _label_graph_check[label] = true;
                    _global_mtx.unlock();
                }
                
            }
            
            {
                // consider old labels
                for (uint32_t ii=1;ii<_graph_row_index[id].size();ii++){
                    uint32_t label = _graph_row_index[id][ii-1].first;
                    uint32_t degree = _graph_row_index[id][ii].second - _graph_row_index[id][ii-1].second;
                    if (degree == 0) continue;
                    if (std::find(labels_need_graph_build.begin(),labels_need_graph_build.end(),label)!=labels_need_graph_build.end()) continue;
                    std::vector<uint32_t> candidates(Lsize,std::numeric_limits<uint32_t>::max());
                    std::vector<float> candidates_distance(Lsize,std::numeric_limits<float>::max());
                    search(data,std::vector<uint32_t>(1,label),Lsize,candidates.data(),candidates_distance.data(),ef_construction);
                    // pruning
                    std::vector<uint32_t> neighbors;
                    neighbors.reserve(degree);
                    std::unordered_map<uint32_t,float> neighbor_to_dis;
                    if (candidates[0]>_nd) continue;
                    neighbors.emplace_back(candidates[0]);
                    uint32_t count = 1;
                    for (uint32_t i=1;i<Lsize;i++){
                        bool prune = false;
                        if (candidates[i]>_nd) break;
                        for (uint32_t neigh:neighbors){
                            float distance2 = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+candidates[i]*_point_size),_dim);
                            if (candidates_distance[i]>distance2*alpha){ // candidate e is closer to neighbor than to q
                                prune=true;
                                break;
                            }
                        }
                        if (!prune){
                            count++;
                            neighbors.push_back(candidates[i]);
                            neighbor_to_dis[candidates[i]]=candidates_distance[i];
                            if (count==degree) break;
                        }
                    }
                    
                    // set neighbors
                    uint32_t lock_id = getLockIndex(id);
                    _label_lock[lock_id].lock();
                    uint32_t start_index = _graph_row_index[id][ii-1].second;
                    memcpy(_storage+id*_point_size+_graph_offset+start_index*sizeof(uint32_t),neighbors.data(),neighbors.size()*sizeof(uint32_t));
                    _label_lock[lock_id].unlock();

                    // insert id into neighbors' neighbor list
                    for (uint32_t neigh:neighbors){
                        // get label neighbor
                        uint32_t index_1 = 0, index_2 = 0;
                        uint32_t new_degree = 0;
                        float dis_id_neigh = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+id*_point_size),_dim);
                        uint32_t neighbor_lock_id = getLockIndex(neigh);
                        _label_lock[neighbor_lock_id].lock();
                        for (uint32_t i=0;i<_graph_row_index[neigh].size();i++){
                            auto& p = _graph_row_index[neigh][i];
                            if (p.first == label){
                                index_1 = p.second;
                                index_2 = _graph_row_index[neigh][i+1].second;
                                break;
                            }
                        }
                        if (index_1==index_2) {
                            _label_lock[neighbor_lock_id].unlock();
                            continue;
                        }
                        {   
                            uint32_t* start_pointer = (uint32_t*)(_storage+neigh*_point_size+_graph_offset);
                            uint32_t* end_pointer = start_pointer+index_2-1;
                            start_pointer+=index_1;
                            uint32_t allocate_degree = index_2 - index_1;
                            if (*end_pointer<_nd){
                                uint32_t end_pointer_id = *end_pointer;
                                float end_pointer_dis = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+end_pointer_id*_point_size),_dim);
                                end_pointer++;
                                std::priority_queue<std::pair<float, uint32_t>> neighbor_pq;
                                neighbor_pq.emplace(std::pair<float,uint32_t>(-dis_id_neigh,id));
                                for (uint32_t *neighbor_pointer = start_pointer; neighbor_pointer < end_pointer; ++neighbor_pointer){
                                    uint32_t neighbor_id = *neighbor_pointer;
                                    if (neighbor_id>_nd) break;
                                    float distance_neigh_nn = _dist_fn->compare((T*)(_storage+neigh*_point_size),(T*)(_storage+neighbor_id*_point_size),_dim);
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
                                        float dis2 = _dist_fn->compare((T*)(_storage+second_pair*_point_size),(T*)(_storage+neighbor_id*_point_size),_dim);
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
                                new_degree = prune_result.size();
                            }
                            else {
                                end_pointer++;
                                for (; start_pointer < end_pointer; ++start_pointer){
                                    if (*start_pointer>=_nd) break;
                                    new_degree++;
                                }
                                *start_pointer = id;
                                new_degree++;
                            }
                            
                        }
                        _label_lock[neighbor_lock_id].unlock();
                    }
                }

            }

            
            for (uint32_t label:labels){
                _global_mtx.lock();
                if (_label_to_pts.find(label)==_label_to_pts.end()){
                    _label_to_pts[label] = std::vector<uint32_t>(1,id);
                    _label_to_medoid_id[label] = std::vector<uint32_t>(1,id);
                    _label_graph_check[label] = false;
                }
                else{
                    _label_to_pts[label].push_back(id);
                    if (_label_to_medoid_id[label].size()<_num_of_entry_point){
                        _label_to_medoid_id[label].push_back(id);
                    }
                    else{
                        uint32_t cur_num = _label_to_pts[label].size();
                        uint32_t rand_ind = rand() % cur_num;
                        if (rand_ind < _num_of_entry_point){
                            _label_to_medoid_id[label][rand_ind] = id;
                        }
                    }
                }
                _global_mtx.unlock();
            }
           
        }
        
        void save(std::string& save_path_prefix){
            {
                std::string bin_file = save_path_prefix+"_storage.bin";
                std::ofstream writer(bin_file,std::ios::out);
                writer.write((char*)&_nd,sizeof(size_t));
                writer.write((char*)&_dim,sizeof(size_t));
                writer.write((char*)&_graph_degree,sizeof(size_t));
                writer.write((char*)_storage,_nd*_point_size);
                writer.close();
            }

            {
                std::string row_index_file = save_path_prefix+"_graph_row_index.bin";
                std::ofstream writer(row_index_file);
                writer.write((char*)&_nd,sizeof(size_t));
                for (uint32_t i=0;i<_nd;i++){
                    std::vector<std::pair<uint32_t, uint32_t>>& row_index = _graph_row_index[i];
                    size_t length = row_index.size();
                    writer.write((char*)&length,sizeof(size_t));
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
                    uint32_t second = p.second;
                    writer.write((char*)&first,sizeof(uint32_t));
                    writer.write((char*)&second,sizeof(uint32_t));
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
                _graph_offset = _dim*sizeof(T);
                _point_size = _graph_offset + _graph_degree*sizeof(uint32_t);
                if (_storage==nullptr){
                    _storage = (char*)malloc(_nd*_point_size);
                }
                reader.read((char*)_storage,_nd*_point_size);
                reader.close();
            }

            {
                std::string row_index_file = save_path_prefix+"_graph_row_index.bin";
                std::ifstream reader(row_index_file);
                reader.read((char*)&_nd,sizeof(size_t));
                _graph_row_index.resize(_nd);
                for (uint32_t i=0;i<_nd;i++){
                    size_t length = 0;
                    reader.read((char*)&length,sizeof(size_t));
                    _graph_row_index[i].resize(length);
                    for (size_t j=0;j<length;j++){
                        uint32_t first,second;
                        reader.read((char*)&first,sizeof(uint32_t));
                        reader.read((char*)&second,sizeof(uint32_t));
                        _graph_row_index[i][j] = std::pair<uint32_t,uint32_t>(first,second);
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
                    size_t label_num;
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
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&label,sizeof(uint32_t));
                    reader.read((char*)&point_num,sizeof(size_t));
                    std::vector<uint32_t> points(point_num,0);
                    reader.read((char*)points.data(),sizeof(uint32_t)*point_num);
                    _label_to_pts[label] = points;
                }
                reader.close();
            }

            {
                std::string hierarchy_file = save_path_prefix+"_hierarchy.bin";
                std::ifstream reader(hierarchy_file);
                size_t label_num;
                uint32_t first,second;
                reader.read((char*)&label_num,sizeof(size_t));
                for (size_t i=0;i<label_num;i++){
                    reader.read((char*)&first,sizeof(uint32_t));
                    reader.read((char*)&second,sizeof(uint32_t));
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