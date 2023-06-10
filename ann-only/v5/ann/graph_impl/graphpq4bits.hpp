    template<class FeatVec_T>
    struct GraphProductQuantizer4Bits : GraphBase {
        typedef FeatVec_T feat_vec_t;
        ProductQuantizer4Bits quantizer;
        index_type num_node;
        // code_dimension is number of 4 bits code used to encode a data point in GraphPQ4Bits
        // code_dimension can be different from parameter num_local_codebooks in quantizer
        // as we might adjust code_dimension to make it divisble by 4. More details can be
        // found in pad_parameters function of ann/quantizer_impl/x86.hpp
        size_t code_dimension;
        // code_offset helps to locate memory position containing neighboring codes
        size_t code_offset;  
        size_t node_mem_size;
        index_type max_degree;
        std::vector<uint64_t> mem_start_of_node;
        std::vector<char> buffer;

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&code_dimension, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&code_offset, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&node_mem_size, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
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
            quantizer.save(fp);
            fclose(fp);
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&code_dimension, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&code_offset, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&node_mem_size, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
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

            quantizer.load(fp);

            fclose(fp);
        }


        void build_quantizer(const pecos::drm_t& X_trn, index_type subspace_dimension, index_type sub_sample_points) {
            size_t code_dimension = X_trn.cols;
            if (subspace_dimension == 0) {
                if (code_dimension >= 400) {
                    code_dimension =  code_dimension % 2 == 0 ? code_dimension / 2 : code_dimension / 2 + 1;
                }
            } else {
                code_dimension = code_dimension / subspace_dimension;
            }
            // currently, we don't support padding 0 on X_trn, so the cols of X_trn must be divisible by subspace_dimension.
            // otherwise, we will throw error in quantizer.train().
            quantizer.train(X_trn, code_dimension, sub_sample_points);
            quantizer.pack_codebook_for_inference();
            this->code_dimension = code_dimension;
        }

        void build_graph(GraphL0<feat_vec_t>& G) {
            max_degree = G.max_degree;
            quantizer.pad_parameters(max_degree, code_dimension);
            num_node = G.num_node;
            size_t num_of_local_centroids = quantizer.num_of_local_centroids;
            size_t neighbor_size = (1 + max_degree) * sizeof(index_type);
            code_offset = neighbor_size;

            std::vector<std::vector<uint8_t>> X_trn_codes(num_node, std::vector<uint8_t> (code_dimension, 0));
            for (size_t i = 0 ; i < num_node ; i++) {
                quantizer.encode(G.get_node_feat(i).val, X_trn_codes[i].data());
            }

            node_mem_size = neighbor_size + max_degree * code_dimension / 2;
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;

            for (size_t i = 0; i < num_node; i++) {
                mem_start_of_node[i + 1] = mem_start_of_node[i] + node_mem_size;
            }

            buffer.resize(mem_start_of_node[num_node], 0);

            for (size_t i = 0; i < num_node; i++) {
                std::vector<std::vector<uint8_t>> neighbor_codes(max_degree, std::vector<uint8_t> (code_dimension, 0));

                memcpy(&buffer[mem_start_of_node[i]], &G.buffer[G.mem_start_of_node[i]], (1 + G.max_degree) * sizeof(index_type));

                index_type size = *reinterpret_cast<index_type*>(&G.buffer[G.mem_start_of_node[i]]);
                for (index_type j = 0; j < size; j++) {
                    index_type member = *reinterpret_cast<index_type *>(&G.buffer[G.mem_start_of_node[i] + sizeof(index_type) + j * sizeof(index_type)]);
                    memcpy(neighbor_codes[j].data(), X_trn_codes[member].data(), code_dimension);
                }

                index_type processed_num_of_neighbors = 0;
                std::vector<char> group_transposed_graph_codes(max_degree / 2 * code_dimension, 0);

                while (processed_num_of_neighbors < size) {
                    std::vector<char> group_code(num_of_local_centroids / 2 * code_dimension, 0);

                    for (index_type k = 0; k < code_dimension; k++) {
                        for (index_type j = 0; j < num_of_local_centroids; j += 2) {
                            uint8_t obj = neighbor_codes[processed_num_of_neighbors + j][k];
                            obj += (neighbor_codes[processed_num_of_neighbors + j + 1][k] << 4);
                            group_code[k * num_of_local_centroids / 2 + j / 2] = obj;
                        }
                    }
                    memcpy(&group_transposed_graph_codes[processed_num_of_neighbors * code_dimension / 2], &group_code[0], num_of_local_centroids * code_dimension / 2);
                    processed_num_of_neighbors += num_of_local_centroids;
                }
                memcpy(&buffer[mem_start_of_node[i] + (1 + max_degree) * sizeof(index_type)], group_transposed_graph_codes.data(), max_degree * code_dimension / 2);
            }
        }

        inline const char* get_neighbor_codes(index_type node_id) const {
            return &buffer[mem_start_of_node[node_id] + code_offset];
        }
        inline const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const {
            const index_type *neighborhood_ptr = nullptr;
            neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer.data()[node_id * (mem_index_type) node_mem_size]);
            return NeighborHood((void*)neighborhood_ptr);
        }
    };
