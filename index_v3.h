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

#define MAX_LOCKS 64
#define getLockIndex(a) a&63
#define MAX_DEGREE 150

std::mutex global_lock;

template<typename T>
class FilterIndex_v3;

template<typename T>
class Index{
    public:
        T* _data = nullptr;
        Distance<T>* _dist_fn = nullptr;
        uint32_t _dim = 0;
        uint32_t _graph_degree = 0;
        uint32_t _ep_num = 0;
        uint32_t _threshold = 0;
        uint32_t _num_of_points = 0;
        bool _graph_index = false;
        std::vector<uint32_t> _ids;
        std::vector<std::vector<uint32_t>>_edge_list;
        std::vector<uint32_t> _entry_points;
        std::mutex _build_lock;
        
    
    Index(T* data, Distance<T>* dist_fn, uint32_t dim, uint32_t graph_degree, uint32_t ep_num, uint32_t threshold){
        _data = data; _dist_fn = dist_fn; _dim = dim; _graph_degree = graph_degree; _ep_num = ep_num; _threshold = threshold;
        _entry_points = std::vector<uint32_t>(_ep_num,0);
    }

    Index(T* data, Distance<T>* dist_fn, uint32_t dim, uint32_t ep_num, uint32_t threshold){
        _data = data; _dist_fn = dist_fn; _dim = dim; _ep_num = ep_num; _threshold = threshold;
        _entry_points = std::vector<uint32_t>(_ep_num,0);
    }

    void save(std::ofstream& writer){
        writer.write((char*)&_graph_degree,sizeof(uint32_t));
        writer.write((char*)&_graph_index,sizeof(bool));
        writer.write((char*)&_num_of_points,sizeof(uint32_t));
        writer.write((char*)_ids.data(),sizeof(uint32_t)*_num_of_points);
        writer.write((char*)_entry_points.data(),sizeof(uint32_t)*_ep_num);
        if (_graph_index){
            for (uint32_t i = 0; i < _num_of_points; i++){
                uint32_t point_deg = _edge_list[i].size();
                writer.write((char*)&point_deg,sizeof(uint32_t));
                writer.write((char*)_edge_list[i].data(),sizeof(uint32_t)*point_deg);
            }
        }
        writer.flush();
    }

    void load(std::ifstream& reader){
        reader.read((char*)&_graph_degree,sizeof(uint32_t));
        reader.read((char*)&_graph_index,sizeof(bool));
        reader.read((char*)&_num_of_points,sizeof(uint32_t));
        _ids = std::vector<uint32_t>(_num_of_points);
        reader.read((char*)_ids.data(),sizeof(uint32_t)*_num_of_points);
        _entry_points = std::vector<uint32_t>(_ep_num);
        reader.read((char*)_entry_points.data(),sizeof(uint32_t)*_ep_num);
        if (_graph_index){
            _edge_list = std::vector<std::vector<uint32_t>>(_num_of_points);
            for (uint32_t i = 0; i < _num_of_points; i++){
                uint32_t point_deg = 0;
                reader.read((char*)&point_deg,sizeof(uint32_t));
                _edge_list[i] = std::vector<uint32_t>(point_deg);
                reader.read((char*)_edge_list[i].data(),sizeof(uint32_t)*point_deg);
            }
        }
    }

    uint32_t search(T* query, uint32_t topk, uint32_t* ids_res, float* dis_res, uint32_t ef_search, bool verbose=false){
        std::priority_queue<std::pair<float,uint32_t>> nearest_neighbor_queue;
        uint32_t computation = 0;
        if (verbose){
            std::cout<<"num_of_point: "<<_num_of_points<<", _graph_index: "<<_graph_index<<std::endl;
        }
        if (_graph_index){
            std::priority_queue<std::pair<float,uint32_t>> candidate_queue;
            std::vector<uint32_t> visit_set;
            for (uint32_t ep: _entry_points){
                uint32_t actual_id = _ids[ep];
                float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                computation++;
                nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,actual_id));
                candidate_queue.push(std::pair<float,uint32_t>(-dis,ep));
                visit_set.push_back(ep);
                if (verbose){
                    std::cout<<"ep: "<<ep<<", actual_id: "<<actual_id<<", dis: "<<dis<<std::endl;
                }
            }
            float lower_bound = std::numeric_limits<float>::max();
            while (!candidate_queue.empty()){
                auto p = candidate_queue.top();
                float target_dis = -p.first;
                uint32_t target = p.second;
                if (target_dis > lower_bound && nearest_neighbor_queue.size() >= ef_search){
                    break;
                }
                candidate_queue.pop();
                if (verbose){
                    std::cout<<"target: "<<target<<", target_dis: "<<target_dis<<", degree: "<<_edge_list[target].size()<<std::endl;
                }
                for (uint32_t neighbor:_edge_list[target]){
                    if (std::find(visit_set.begin(),visit_set.end(),neighbor)==visit_set.end()){
                        uint32_t actual_id = _ids[neighbor];
                        float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                        computation++;
                        visit_set.push_back(neighbor);
                        if (nearest_neighbor_queue.size() < ef_search || lower_bound > dis){
                            candidate_queue.push(std::pair<float,uint32_t>(-dis,neighbor));
                            nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,actual_id));
                            if (nearest_neighbor_queue.size() > ef_search){
                                nearest_neighbor_queue.pop();
                                lower_bound = nearest_neighbor_queue.top().first;
                            }
                        }
                        if (verbose){
                            std::cout<<"Neighbor: "<<neighbor<<", actual id: "<<actual_id<<", dis: "<<dis<<std::endl;
                        }
                    }
                    else if (verbose){
                        std::cout<<"Neighbor visited: "<<neighbor<<", actual id: "<<_ids[neighbor]<<std::endl;
                    }
                }
            }
        }
        else{
            computation+=_ids.size();
            for (uint32_t candidate: _ids){
                float dis = _dist_fn->compare(query,_data+candidate*_dim,_dim);
                nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,candidate));
                if (verbose){
                    std::cout<<"Brute-force: candidate = "<<candidate<<", distance = "<<dis<<std::endl;
                }
                if (nearest_neighbor_queue.size()>topk){
                    nearest_neighbor_queue.pop();
                }
            }
        }
        while (nearest_neighbor_queue.size() > topk){
            nearest_neighbor_queue.pop();
        }
        uint32_t position = nearest_neighbor_queue.size();
        while (!nearest_neighbor_queue.empty()){
            position--;
            std::pair<float,uint32_t> p = nearest_neighbor_queue.top();
            nearest_neighbor_queue.pop();
            ids_res[position] = p.second;
            dis_res[position] = p.first;
        }
        return computation;
    }

    inline bool isPowerOfTwo(uint32_t n) {
        return !(n & (n - 1));
    }

    void _search_and_set_neighbors(uint32_t internal_id, uint32_t id, uint32_t efConstruct, float alpha, bool verbose){
        if (_graph_index && _edge_list[internal_id].size() >= _graph_degree){
            _edge_list[internal_id].resize(_graph_degree);
            return;
        }
        std::vector<uint32_t> neighbor_candidates(efConstruct,std::numeric_limits<uint32_t>::max());
        std::vector<float> neighbor_candidates_distance(efConstruct,-1);
        {
            std::priority_queue<std::pair<float,uint32_t>> nearest_neighbor_queue;
            T* query = _data+id*_dim;
            if (_graph_index){
                std::priority_queue<std::pair<float,uint32_t>> candidate_queue;
                std::vector<uint32_t> visit_set;
                if (_edge_list[internal_id].size()>_entry_points.size()){
                    _build_lock.lock();
                    std::vector<uint32_t> neighbor_list(_edge_list[internal_id]);
                    _build_lock.unlock();
                    for (uint32_t ep: neighbor_list){
                        uint32_t actual_id = _ids[ep];
                        float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                        nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,ep));
                        candidate_queue.push(std::pair<float,uint32_t>(-dis,ep));
                        visit_set.push_back(ep);
                    }
                }
                else{
                    for (uint32_t ep: _entry_points){
                        uint32_t actual_id = _ids[ep];
                        float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                        nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,ep));
                        candidate_queue.push(std::pair<float,uint32_t>(-dis,ep));
                        visit_set.push_back(ep);
                    }
                }
                float lower_bound = std::numeric_limits<float>::max();
                while (!candidate_queue.empty()){
                    auto p = candidate_queue.top();
                    float target_dis = -p.first;
                    uint32_t target = p.second;
                    if (target_dis > lower_bound && nearest_neighbor_queue.size() >= efConstruct){
                        break;
                    }
                    candidate_queue.pop();
                    _build_lock.lock();
                    std::vector<uint32_t> neighbor_list(_edge_list[target]);
                    _build_lock.unlock();

                    for (uint32_t neighbor:neighbor_list){
                        if (std::find(visit_set.begin(),visit_set.end(),neighbor)==visit_set.end()){
                            uint32_t actual_id = _ids[neighbor];
                            float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                            visit_set.push_back(neighbor);
                            if (nearest_neighbor_queue.size() < efConstruct || lower_bound > dis){
                                candidate_queue.push(std::pair<float,uint32_t>(-dis,neighbor));
                                nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,neighbor));
                                if (nearest_neighbor_queue.size() > efConstruct){
                                    nearest_neighbor_queue.pop();
                                    lower_bound = nearest_neighbor_queue.top().first;
                                }
                            }
                        }
                    }
                }
            }
            else{
                for (uint32_t i = 0; i < _num_of_points; i++){
                    uint32_t candidate = _ids[i];
                    float dis = _dist_fn->compare(query,_data+candidate*_dim,_dim);
                    nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,i));
                    if (nearest_neighbor_queue.size()>efConstruct){
			            nearest_neighbor_queue.pop();
		            }
		        }
            }
            while (nearest_neighbor_queue.size() > efConstruct){
                nearest_neighbor_queue.pop();
            }
            uint32_t position = nearest_neighbor_queue.size();
            while (!nearest_neighbor_queue.empty()){
                position--;
                std::pair<float,uint32_t> p = nearest_neighbor_queue.top();
                nearest_neighbor_queue.pop();
                neighbor_candidates[position] = p.second;
                neighbor_candidates_distance[position] = p.first;
            }        
        }
        std::vector<uint32_t> new_neighbors;
        for (uint32_t i = 0; i < efConstruct; i++){
            if (neighbor_candidates[i]>=_num_of_points) break;
            uint32_t candidate = neighbor_candidates[i];
            float dis_cand_target = neighbor_candidates_distance[i];
            uint32_t candidate_actual_id = _ids[candidate];
            if (candidate == internal_id) continue;
            bool prune = false;
            for (uint32_t neighbor: new_neighbors){
                uint32_t neigh_actual_id = _ids[neighbor];
                float dis_cand_neigh = _dist_fn->compare(_data+candidate_actual_id*_dim,_data+neigh_actual_id*_dim,_dim);
                if (dis_cand_neigh * alpha < dis_cand_target){
                    prune = true;
                    break;
                }
            }
            if (!prune){
                new_neighbors.push_back(candidate);
            }
            if (new_neighbors.size() >= _graph_degree) break;
        }
        _build_lock.lock();
        std::vector<uint32_t>().swap(_edge_list[internal_id]);
        _edge_list[internal_id].resize(new_neighbors.size());
        memcpy(_edge_list[internal_id].data(),new_neighbors.data(),new_neighbors.size()*sizeof(uint32_t));
        _build_lock.unlock();

        for (uint32_t target: new_neighbors){
            uint32_t target_actual_id = _ids[target];
            if (_edge_list[target].size() < _graph_degree){
                _build_lock.lock();
                _edge_list[target].push_back(internal_id);
                _build_lock.unlock();
            }
            else{
                // pruning
                std::priority_queue<std::pair<float,uint32_t>> candidate_queue;
                _build_lock.lock();
                std::vector<uint32_t> copy_target_neighbors(_edge_list[target]);
                _build_lock.unlock();
                for (uint32_t neighbor: copy_target_neighbors){
                    uint32_t neigh_actual_id = _ids[neighbor];
                    float dis = _dist_fn->compare(_data+target_actual_id*_dim,_data+neigh_actual_id*_dim,_dim);
                    candidate_queue.push(std::pair<float,uint32_t>(-dis,neighbor));
                }
                float dis = _dist_fn->compare(_data+target_actual_id*_dim,_data+id*_dim,_dim);
                candidate_queue.push(std::pair<float,uint32_t>(-dis,internal_id));
                std::vector<uint32_t> prune_result;
                while (!candidate_queue.empty()){
                    uint32_t candidate = candidate_queue.top().second;
                    float dis_cand_target = -candidate_queue.top().first;
                    uint32_t candidate_actual_id = _ids[candidate];
                    candidate_queue.pop();
                    bool prune = false;
                    for (uint32_t neighbor:prune_result){
                        uint32_t neigh_actual_id = _ids[neighbor];
                        float dis_cand_neigh = _dist_fn->compare(_data+candidate_actual_id*_dim,_data+neigh_actual_id*_dim,_dim);
                        if (dis_cand_neigh * alpha < dis_cand_target){
                            prune = true;
                            break;
                        }
                    }
                    if (!prune){
                        prune_result.push_back(candidate);
                    }
                    if (prune_result.size() >= _graph_degree) break;
                }
                _build_lock.lock();
                _edge_list[target].resize(prune_result.size());
                memcpy(_edge_list[target].data(),prune_result.data(),sizeof(uint32_t)*prune_result.size());
                _build_lock.unlock();
            }
        }
    }

    uint32_t insert(uint32_t id, uint32_t efConstruct, uint32_t& len, uint32_t bigM, float alpha = 1.0, bool verbose = false){
        bool rebuild = false;
        _build_lock.lock();
        _ids.push_back(id);
        uint32_t internal_id = _num_of_points;
        _edge_list.push_back(std::vector<uint32_t>());
        _num_of_points++;
        uint32_t cur_snapshot_size = _num_of_points;
        if (_num_of_points < _threshold){
            _build_lock.unlock();
            return bigM;
        }
        else if (_num_of_points == _threshold){
            uint32_t l_t = std::ceil(std::log2(_num_of_points));
            len += l_t;
            rebuild = true;
            _graph_degree = std::min(MAX_DEGREE,(int)std::ceil((float)bigM*(float)l_t/(float)len));
            for (uint32_t i = 0; i < _ep_num; i++){
                _entry_points[i] = rand() % cur_snapshot_size;
            }
            if (verbose){
                std::cout<<"trigger threshold, nd="<<_num_of_points<<", lt="<<l_t<<", len="<<len<<", allocated_deg="<<_graph_degree<<std::endl;
            }
        }
        else if (isPowerOfTwo(_num_of_points)){
            uint32_t l_t = std::ceil(std::log2(_num_of_points));
            len++;
            uint32_t new_graph_degree = std::min(MAX_DEGREE,(int)std::ceil((float)bigM*(float)l_t/(float)len));
            if (new_graph_degree > alpha * _graph_degree || new_graph_degree < _graph_degree){
                _graph_degree = new_graph_degree;
                rebuild = true;
            }
            if (verbose){
                std::cout<<"trigger power of two, nd="<<_num_of_points<<", lt="<<l_t<<", len="<<len<<", allocated_deg="<<_graph_degree<<std::endl;
            }
        }
        _build_lock.unlock();
        efConstruct = std::max(efConstruct,_graph_degree);
        if (rebuild){
            for (uint32_t i = 0; i < cur_snapshot_size; i++){
                _search_and_set_neighbors(i, _ids[i], efConstruct, alpha, verbose);
            }
            _graph_index = true;
        }
        else{
            _search_and_set_neighbors(internal_id, id, efConstruct, alpha, verbose);
        }
        uint32_t l_t = std::ceil(std::log2(cur_snapshot_size));
        return l_t;
    }

    void rebuild(uint32_t efConstruct, float alpha, bool verbose){
        for (uint32_t i = 0; i < _num_of_points; i++){
            _search_and_set_neighbors(i, _ids[i], efConstruct, alpha, verbose);
        }
        _graph_index = true;
    }

    uint32_t search_and(T* query, std::vector<uint32_t>& labels, bool (FilterIndex_v3<T>::*match_func)(uint32_t,const std::vector<uint32_t>&), 
                        FilterIndex_v3<T>& instance, uint32_t topk, uint32_t* ids_res, float* dis_res, uint32_t ef_search, bool verbose = false){
        std::priority_queue<std::pair<float,uint32_t>> nearest_neighbor_queue;
        std::priority_queue<std::pair<float,uint32_t>> result_pq;
        uint32_t computation = 0;
        if (verbose){
            std::cout<<"num_of_point: "<<_num_of_points<<", _graph_index: "<<_graph_index<<std::endl;
        }
        if (_graph_index){
            std::priority_queue<std::pair<float,uint32_t>> candidate_queue;
            std::vector<uint32_t> visit_set;
            for (uint32_t ep: _entry_points){
                uint32_t actual_id = _ids[ep];
                float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                computation++;
                nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,actual_id));
                candidate_queue.push(std::pair<float,uint32_t>(-dis,ep));
                visit_set.push_back(ep);
                if ((instance.*match_func)(actual_id,labels)){
                    result_pq.push(std::pair<float,uint32_t>(dis,actual_id));
                }
                if (verbose){
                    std::cout<<"ep: "<<ep<<", actual_id: "<<actual_id<<", dis: "<<dis<<std::endl;
                }
            }
            float lower_bound = std::numeric_limits<float>::max();
            while (!candidate_queue.empty()){
                auto p = candidate_queue.top();
                float target_dis = -p.first;
                uint32_t target = p.second;
                if (target_dis > lower_bound && nearest_neighbor_queue.size() >= ef_search){
                    break;
                }
                candidate_queue.pop();
                if (verbose){
                    std::cout<<"target: "<<target<<", target_dis: "<<target_dis<<", degree: "<<_edge_list[target].size()<<std::endl;
                }
                for (uint32_t neighbor:_edge_list[target]){
                    if (std::find(visit_set.begin(),visit_set.end(),neighbor)==visit_set.end()){
                        uint32_t actual_id = _ids[neighbor];
                        float dis = _dist_fn->compare(query,_data+actual_id*_dim,_dim);
                        computation++;
                        visit_set.push_back(neighbor);
                        if ((instance.*match_func)(actual_id,labels)){
                            result_pq.push(std::pair<float,uint32_t>(dis,actual_id));
                        }
                        if (nearest_neighbor_queue.size() < ef_search || lower_bound > dis){
                            candidate_queue.push(std::pair<float,uint32_t>(-dis,neighbor));
                            nearest_neighbor_queue.push(std::pair<float,uint32_t>(dis,actual_id));
                            if (nearest_neighbor_queue.size() > ef_search){
                                nearest_neighbor_queue.pop();
                                lower_bound = nearest_neighbor_queue.top().first;
                            }
                        }
                        if (verbose){
                            std::cout<<"Neighbor: "<<neighbor<<", actual id: "<<actual_id<<", dis: "<<dis<<std::endl;
                        }
                    }
                    else if (verbose){
                        std::cout<<"Neighbor visited: "<<neighbor<<", actual id: "<<_ids[neighbor]<<std::endl;
                    }
                }
            }
        }
        else{
            computation+=_ids.size();
            std::vector<uint32_t> candidate_list;
            for (uint32_t candidate: _ids){
                if ((instance.*match_func)(candidate,labels)){
                    float dis = _dist_fn->compare(query,_data+candidate*_dim,_dim);
                    if (verbose){
                        std::cout<<"Brute-force: candidate = "<<candidate<<", distance = "<<dis<<std::endl;
                    }
                    result_pq.push(std::pair<float,uint32_t>(dis,candidate));
                    if (result_pq.size()>topk){
                        result_pq.pop();
                    }
                }
            }
        }
        while (result_pq.size() > topk){
            result_pq.pop();
        }
        uint32_t position = result_pq.size();
        if (verbose){
            std::cout<<"find "<<position<<" neighbors"<<std::endl;
        }
        while (!result_pq.empty()){
            position--;
            std::pair<float,uint32_t> p = result_pq.top();
            result_pq.pop();
            ids_res[position] = p.second;
            dis_res[position] = p.first;
        }
        return computation;            
    }

    void build(std::vector<uint32_t>& ids, uint32_t efConstruct, uint32_t degree, float alpha = 1.0f, bool verbose = false){
        _graph_degree = degree;
        _num_of_points = ids.size();
        _ids.resize(_num_of_points);
        memcpy(_ids.data(),ids.data(),sizeof(uint32_t)*_num_of_points);
        _edge_list.resize(_num_of_points);
        if (degree == 0) return;
        
        #pragma omp parallel for
        for (uint32_t i = 0; i < _threshold; i++){
            std::priority_queue<std::pair<float,uint32_t>> pq;
            uint32_t i_id = ids[i];
            for (uint32_t j = 0; j < _threshold; j++){
                if (i==j) continue;
                uint32_t j_id = ids[j];
                float dis = _dist_fn->compare((T*)(_data+i_id*_dim),(T*)(_data+j_id*_dim),_dim);
                pq.push(std::pair<float,uint32_t>(-dis,j));
            }
            std::vector<uint32_t> new_neighbors;
            while (!pq.empty()){
                auto p = pq.top();
                pq.pop();
                bool prune = false;
                uint32_t dis_target_cand = -p.first;
                uint32_t cand = p.second;
                for (uint32_t neigh:new_neighbors){
                    float dis_neigh_cand = _dist_fn->compare((T*)(_data+_ids[cand]*_dim),(T*)(_data+_ids[neigh]*_dim),_dim);
                    if (dis_target_cand * alpha < dis_neigh_cand){
                        prune = true;
                        break;
                    }
                }
                if (!prune){
                    new_neighbors.push_back(cand);
                    if (new_neighbors.size() >= degree) break;
                }
            }
            _build_lock.lock();
            _edge_list[i].resize(new_neighbors.size());
            memcpy(_edge_list[i].data(),new_neighbors.data(),sizeof(uint32_t)*new_neighbors.size());
            _build_lock.unlock();
        }
    
        for (uint32_t i = 0; i < _ep_num; i++){
            while (true){
                uint32_t rand_index = rand() % _threshold;
                if (std::find(_entry_points.begin(),_entry_points.end(),rand_index)==_entry_points.end()){
                    _entry_points.push_back(rand_index);
                    break;
                }
            }
        }

        _graph_index = true;
        #pragma omp parallel for
        for (uint32_t i = _threshold; i < _num_of_points; i++){
            _search_and_set_neighbors(i,_ids[i],efConstruct,alpha,verbose);
        }
    }

};

template<typename T>
class FilterIndex_v3{
    public:
        T* _data = nullptr;
        Distance<T>* _dist_fn = nullptr;
        uint32_t _nd = 0;
        uint32_t _dim = 0;
        uint32_t _len = 0;
        uint32_t _degree_budget = 0;
        uint32_t _ep_num = 0;
        uint32_t _threshold = 0;
        std::unordered_map<uint32_t, Index<T>*> tag2index;
        std::vector<std::vector<uint32_t>> _pts_to_labels;
        std::mutex _lock;
    
    FilterIndex_v3(uint32_t nd, uint32_t dim, uint32_t degree_budget, Distance<T>* dist_fn, uint32_t ep_num, uint32_t threshold){
        _nd = nd; _dim = dim; _degree_budget = degree_budget; _dist_fn = dist_fn;
        _data = (T*)malloc(sizeof(T)*_nd*_dim);
        _len = 0;
        _ep_num = ep_num; _threshold = threshold;
        _pts_to_labels.resize(_nd);
    }

    FilterIndex_v3(Distance<T>* dist_fn, std::string index_path){
        _dist_fn = dist_fn;
        std::ifstream reader(index_path);
        reader.read((char*)&_nd,sizeof(uint32_t));
        reader.read((char*)&_dim,sizeof(uint32_t));
        _data = (T*)malloc(sizeof(T)*_nd*_dim);
        reader.read((char*)_data,sizeof(T)*_nd*_dim);
        reader.read((char*)&_len,sizeof(uint32_t));
        reader.read((char*)&_degree_budget,sizeof(uint32_t));
        reader.read((char*)&_ep_num,sizeof(uint32_t));
        reader.read((char*)&_threshold,sizeof(uint32_t));
        _pts_to_labels.resize(_nd);
        std::cout<<_nd<<", "<<_dim<<", "<<_len<<", "<<_degree_budget<<", "<<_ep_num<<", "<<_threshold<<std::endl;
        uint32_t tag_num = 0;
        reader.read((char*)&tag_num,sizeof(uint32_t));
        std::cout<<"tag_num: "<<tag_num<<std::endl;
        for (uint32_t i = 0; i < tag_num; i++){
            uint32_t tag = 0;
            Index<T>* index = new Index<T>(_data,_dist_fn,_dim,_ep_num,_threshold);
            reader.read((char*)&tag,sizeof(uint32_t));
            index->load(reader);
            tag2index[tag] = index;
            for (uint32_t actual_id: index->_ids){
                _pts_to_labels[actual_id].push_back(tag);
            }
        }
        reader.close();
        for (uint32_t i = 0; i < _nd; i++){
            std::sort(_pts_to_labels[i].begin(),_pts_to_labels[i].end());
        }
    }

    ~FilterIndex_v3(){
        free(_data);
        for (auto p: tag2index){
            delete p.second;
        }
    }

    bool match_all_filters(uint32_t point_id, const std::vector<uint32_t> &incoming_labels){
        auto &curr_node_labels = _pts_to_labels[point_id];
        auto cur_pointer = curr_node_labels.begin();
        for (uint32_t label:incoming_labels){
            cur_pointer = std::find(cur_pointer,curr_node_labels.end(),label);
            if (cur_pointer==curr_node_labels.end()) {
                return false;
            }
        }
        return true;
    }

    void save(std::string index_path){
        std::ofstream writer(index_path);
        writer.write((char*)&_nd,sizeof(uint32_t));
        writer.write((char*)&_dim,sizeof(uint32_t));
        writer.write((char*)_data,sizeof(T)*_nd*_dim);
        writer.write((char*)&_len,sizeof(uint32_t));
        writer.write((char*)&_degree_budget,sizeof(uint32_t));
        writer.write((char*)&_ep_num,sizeof(uint32_t));
        writer.write((char*)&_threshold,sizeof(uint32_t));
        uint32_t tag_num = tag2index.size();
        writer.write((char*)&tag_num,sizeof(uint32_t));
        std::cout<<"tag num: "<<tag_num<<", tag2index size: "<<tag2index.size()<<std::endl;
        for (auto p:tag2index){
            uint32_t tag = p.first;
            Index<T>* index = p.second;
            writer.write((char*)&tag,sizeof(uint32_t));
            index->save(writer);
        }
        writer.close();
    }

    void insert(T* point, uint32_t id, std::vector<uint32_t>& tags, uint32_t efConstruct, float alpha = 1.0f, bool verbose = false){
        bool global_rebuild = false;
        uint32_t unsatisfied_tag = -1;
        memcpy(_data+id*_dim,point,sizeof(T)*_dim);
        _pts_to_labels[id] = tags;
        for (uint32_t tag: tags){
            global_lock.lock();
            Index<T>* index = nullptr;
            if (tag2index.find(tag)==tag2index.end()){
                index = new Index<T>(_data,_dist_fn,_dim,_ep_num,_threshold);
                tag2index[tag] = index;
            }
            if (verbose){
                std::cout<<"Insert "<<id<<" to index of tag "<<tag<<std::endl;
            }
            index = tag2index[tag];
            global_lock.unlock();
            uint32_t l_t = index->insert(id,efConstruct,_len,_degree_budget,alpha,verbose);
            if (2*_degree_budget*l_t < _len){
                global_rebuild = true;
                std::cout<<"Global Rebuild Needed"<<std::endl;
                std::cout<<_degree_budget<<", "<<tag<<", "<<l_t<<", "<<index->_num_of_points<<", "<<_len<<", "<<index->_threshold<<std::endl;
            }
        }
        if (global_rebuild){
            std::cout<<"GLOBAL REBUILD"<<std::endl;
            global_lock.lock();
            while (true){
                bool terminate = true;
                _threshold = 2*_threshold;
                _len = 0;
                for (auto p: tag2index){
                    uint32_t tag = p.first;
                    uint32_t cardinality = p.second->_num_of_points;
                    p.second->_threshold = _threshold;
                    if (cardinality >= _threshold){
                        uint32_t l_t = std::ceil(std::log2(cardinality));
                        _len += l_t;
                    }
                }
                for (auto p: tag2index){
                    uint32_t cardinality = p.second->_num_of_points;
                    if (cardinality >= _threshold){
                        uint32_t l_t = std::ceil(std::log2(cardinality));
                        uint32_t deg = (float)l_t*(float)_degree_budget/(float)_len;
                        p.second->_graph_degree = deg;
                        if (2*_degree_budget*l_t < _len){
                            terminate = false;
                        }
                    }
                }
                if (terminate) break;
            }
            for (auto p: tag2index){
                uint32_t cardinality = p.second->_num_of_points;
                if (cardinality >= _threshold){
                    p.second->rebuild(efConstruct,alpha,verbose);
                }
            }
            global_lock.unlock();
        }
    }

    uint32_t search(T* query, uint32_t tag, uint32_t topk, uint32_t efSearch, uint32_t* res_ids, float* res_dis, bool verbose = false){
        if (tag2index.find(tag)!=tag2index.end()){
            return tag2index[tag]->search(query,topk,res_ids,res_dis,efSearch,verbose);
        }
        else{
            return 0;
        }
    }

    uint32_t search_predicate(T* query, std::vector<uint32_t>& labels, uint32_t topk, uint32_t efSearch, uint32_t* res_ids, float* res_dis, bool verbose = false){
        uint32_t cardinality = _nd;
        uint32_t smallest_cardinality = _nd;
        uint32_t smallest_filter = labels[0];
        std::sort(labels.begin(),labels.end());
        bool smallest_graph_search = true;
        {
            std::vector<float> label_size;
            for (uint32_t label:labels){
                global_lock.lock();
                if (tag2index.find(label)==tag2index.end()){
                    std::cout<<"Cannot find label "<<label<<std::endl;
                    exit(-1);
                }
                uint32_t current_label_size = tag2index[label]->_num_of_points;
                label_size.push_back(current_label_size);
                if (current_label_size<smallest_cardinality){
                    smallest_cardinality = current_label_size;
                    smallest_filter = label;
                    smallest_graph_search = tag2index[label]->_graph_index;
                }
                global_lock.unlock();
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
        global_lock.lock();
        Index<T>* smallest_index = tag2index[smallest_filter];
        global_lock.unlock();
        if (verbose){
            std::cout<<"smallest filter: "<<smallest_filter<<", estimated cardinality: "<<cardinality<<std::endl;
        }
        return smallest_index->search_and(query,labels,&FilterIndex_v3<T>::match_all_filters,*this,topk,res_ids,res_dis,efSearch,verbose);
    }


};
