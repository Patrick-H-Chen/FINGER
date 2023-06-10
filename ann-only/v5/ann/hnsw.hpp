/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ann/feat_vectors.hpp"
#include "ann/quantizer.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "utils/file_util.hpp"
#include "utils/matrix.hpp"
#include "utils/random.hpp"
#include "utils/type_util.hpp"

namespace pecos {

namespace ann {

    typedef uint32_t index_type;
    typedef uint64_t mem_index_type;

    struct NeighborHood {
        index_type* degree_ptr;
        index_type* neighbor_ptr;

        NeighborHood(void *memory_ptr) {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            degree_ptr = reinterpret_cast<index_type*>(curr_ptr);
            curr_ptr += sizeof(index_type);
            neighbor_ptr = reinterpret_cast<index_type*>(curr_ptr);
        }

        void set_degree(index_type degree) {
            *degree_ptr = degree;
        }

        const index_type& degree() const {
            return *degree_ptr;
        }

        index_type* begin() { return neighbor_ptr; }
        const index_type* begin() const { return neighbor_ptr; }

        index_type* end() { return neighbor_ptr + degree(); }
        const index_type* end() const { return neighbor_ptr + degree(); }

        index_type& operator[](size_t i) { return neighbor_ptr[i]; }
        const index_type& operator[](size_t i) const { return neighbor_ptr[i]; }

        void push_back(index_type dst) {
            neighbor_ptr[*degree_ptr] = dst;
            *degree_ptr += 1;
        }
        void clear() {
            *degree_ptr = 0;
        }
    };

    struct GraphBase {
        virtual const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const = 0;

        NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) {
            return const_cast<const GraphBase&>(*this).get_neighborhood(node_id, dummy_level_id);
        }
    };

    template<class FeatVec_T>
    struct GraphL0 : GraphBase {
        typedef FeatVec_T feat_vec_t;
        index_type num_node;
        index_type feat_dim;
        index_type max_degree;
        index_type node_mem_size;
        std::vector<uint64_t> mem_start_of_node;
        std::vector<char> buffer;

        size_t neighborhood_memory_size() const { return (1 + max_degree) * sizeof(index_type); }

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&feat_dim, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&node_mem_size, 1, fp);
            size_t sz = mem_start_of_node.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            sz = buffer.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<char>(&buffer[0], sz, fp);
            }
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&feat_dim, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&node_mem_size, 1, fp);
            size_t sz = 0;
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            mem_start_of_node.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            buffer.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<char>(&buffer[0], sz, fp);
            }
        }

        template<class MAT_T>
        void init(const MAT_T& feat_mat, index_type max_degree) {
            this->num_node = feat_mat.rows;
            this->feat_dim = feat_mat.cols;
            this->max_degree = max_degree;
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;
            for (size_t i = 0; i < num_node; i++) {
                const feat_vec_t& xi(feat_mat.get_row(i));
                mem_start_of_node[i + 1] = mem_start_of_node[i] + neighborhood_memory_size() + xi.memory_size();
            }
            buffer.resize(mem_start_of_node[num_node], 0);
            if (feat_vec_t::is_fixed_size::value) {
                node_mem_size = buffer.size() / num_node;
            }

            // get_node_feat_ptr must appear after memory allocation (buffer.resize())
            for (size_t i = 0; i < num_node; i++) {
                const feat_vec_t& xi(feat_mat.get_row(i));
                xi.copy_to(get_node_feat_ptr(i));
            }
        }

        inline feat_vec_t get_node_feat(index_type node_id) const {
            return feat_vec_t(const_cast<void*>(get_node_feat_ptr(node_id)));
        }

        inline void prefetch_node_feat(index_type node_id) const {
#ifdef USE_SSE
             _mm_prefetch((char*)get_node_feat_ptr(node_id), _MM_HINT_T0);
#elif defined(__GNUC__)
             __builtin_prefetch((char*)get_node_feat_ptr(node_id), 0, 0);
#endif
        }

        inline const void* get_node_feat_ptr(index_type node_id) const {
            if (feat_vec_t::is_fixed_size::value) {
                return &buffer[node_id * (mem_index_type) node_mem_size + neighborhood_memory_size()];
            } else {
                return &buffer[mem_start_of_node[node_id] + neighborhood_memory_size()];
            }
        }

        inline void* get_node_feat_ptr(index_type node_id) {
            return const_cast<void*>(const_cast<const GraphL0<FeatVec_T>&>(*this).get_node_feat_ptr(node_id));
        }

        inline const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const {
            const index_type *neighborhood_ptr = nullptr;
            if (feat_vec_t::is_fixed_size::value) {
                neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer.data()[node_id * (mem_index_type) node_mem_size]);
            } else {
                neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer[mem_start_of_node[node_id]]);
            }
            return NeighborHood((void*)neighborhood_ptr);
        }
    };

    struct GraphL1 : GraphBase {
        index_type num_node;
        index_type max_level;
        index_type max_degree;
        index_type node_mem_size;
        index_type level_mem_size;
        std::vector<index_type> buffer;

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&node_mem_size, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&level_mem_size, 1, fp);
            size_t sz = buffer.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<index_type>(&buffer[0], sz, fp);
            }
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&node_mem_size, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&level_mem_size, 1, fp);
            size_t sz = 0;
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            buffer.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<index_type>(&buffer[0], sz, fp);
            }
        }

        template<class MAT_T>
        void init(const MAT_T& feat_mat, index_type max_degree, index_type max_level) {
            this->num_node = feat_mat.rows;
            this->max_level = max_level;
            this->max_degree = max_degree;
            this->level_mem_size = 1 + max_degree;
            this->node_mem_size = max_level * this->level_mem_size;
            buffer.resize(num_node * (mem_index_type) this->node_mem_size, 0);
        }

        inline const NeighborHood get_neighborhood(index_type node_id, index_type level_id=0) const {
            const index_type *neighborhood_ptr = &buffer[node_id * (mem_index_type) this->node_mem_size + (level_id - 1) * (mem_index_type) this->level_mem_size];
            return NeighborHood((void*)neighborhood_ptr);
        }
    };

#include "ann/graph_impl/graphfinger.hpp"
#include "ann/graph_impl/graphpq4bits.hpp"

    template<class T>
    struct SetOfVistedNodes {
        T init_token, curr_token;
        std::vector<T> buffer;
        SetOfVistedNodes(int num_nodes) :
            init_token(0),
            curr_token(init_token),
            buffer(num_nodes, T(init_token)) { }

        void mark_visited(unsigned node_id) { buffer[node_id] = curr_token; }

        bool is_visited(unsigned node_id) { return buffer[node_id] == curr_token; }

        // need to reset() for every new query search
        // amortized time complexity is O(num_nodes / std::numeric_limits<T>::max())
        void reset() {
            curr_token += 1;
            if (curr_token == init_token) {
                std::fill(buffer.begin(), buffer.end(), init_token);
                curr_token = init_token + 1;
            }
        }
    };

    template <typename T1, typename T2>
    struct Pair {
        T1 dist;
        T2 node_id;
        Pair(const T1& dist=T1(), const T2& node_id=T2()): dist(dist), node_id(node_id) {}
        bool operator<(const Pair<T1, T2>& other) const { return dist < other.dist; }
        bool operator>(const Pair<T1, T2>& other) const { return dist > other.dist; }
    };

    template<typename T, typename _Compare = std::less<T>>
    struct heap_t : public std::vector<T> {
        typedef typename std::vector<T> container_type;
        typedef typename container_type::value_type value_type;
        typedef typename container_type::reference reference;
        typedef typename container_type::const_reference const_reference;
        typedef typename container_type::size_type size_type;
        typedef _Compare value_compare;

        _Compare comp;

        const_reference top() const { return this->front(); }

        void push(const value_type& __x) {
            this->push_back(__x);
            std::push_heap(this->begin(), this->end(), comp);
        }

#if __cplusplus >= 201103L
        void push(value_type&& __x) {
            this->push_back(std::move(__x));
            std::push_heap(this->begin(), this->end(), comp);
        }

        template<typename... _Args>
        void emplace(_Args&&... __args) {
            this->emplace_back(std::forward<_Args>(__args)...);
            std::push_heap(this->begin(), this->end(), comp);
        }
#endif
        void pop() {
            std::pop_heap(this->begin(), this->end(), comp);
            this->pop_back();
        }
    };
#include "search_struct_impl/hnsw.hpp"
#include "search_struct_impl/hnswpq4bit.hpp"
#include "search_struct_impl/hnswfinger.hpp"

}  // end of namespace ann
}  // end of namespace pecos
