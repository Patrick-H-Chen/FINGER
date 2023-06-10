#include <random>
    template<typename dist_t, class FeatVec_T>
    struct GraphFinger : GraphBase {
        typedef FeatVec_T feat_vec_t;
        Finger<dist_t> finger;
        index_type num_node;
        // code_dimension is number of 4 bits code used to encode a data point in GraphPQ4Bits
        // code_dimension can be different from parameter num_local_codebooks in quantizer
        // as we might adjust code_dimension to make it divisble by 4. More details can be
        // found in pad_parameters function of ann/quantizer_impl/x86.hpp
        // code_offset helps to locate memory position containing neighboring codes
        size_t code_offset;  
        size_t node_mem_size;
        index_type max_degree;
        std::vector<uint64_t> mem_start_of_node;
        std::vector<char> buffer;

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
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
            finger.save(fp);
            //fclose(fp);
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
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

            finger.load(fp);

            //fclose(fp);
        }

        inline void prefetch_node_feat(index_type node_id) const {
#ifdef USE_SSE

            const auto addr = (char*)get_node_feat_ptr(node_id); 
            size_t ptr = 0;
            while (ptr < node_mem_size) {

                _mm_prefetch(addr, _MM_HINT_T0);
                ptr += 64;
            } 

            //for (int r = 8; r < 16; r++) { 
            //    _mm_prefetch(stored_info + r * 64, _MM_HINT_T0);
            //} 
             //_mm_prefetch((char*)get_node_feat_ptr(node_id), _MM_HINT_T0);
#elif defined(__GNUC__)
             __builtin_prefetch((char*)get_node_feat_ptr(node_id), 0, 0);
#endif
        }
        inline const void* get_node_feat_ptr(index_type node_id) const {
            if (feat_vec_t::is_fixed_size::value) {
                return &buffer[node_id * (mem_index_type) node_mem_size];
            } else {
                return &buffer[mem_start_of_node[node_id] + code_offset];
            }
        }



        __attribute__((__target__("default")))
        void pad_parameters() {
        }

        __attribute__((__target__("avx512f"))) 
        void pad_parameters() {
            max_degree = max_degree % 16 == 0 ? max_degree : (max_degree / 16 + 1) * 16;
        }

        void build_graph(const GraphL0<feat_vec_t>& G, int low_rank=64) {
            std::random_device rd;
            std::mt19937 gen(rd());
            const int dimension = G.feat_dim;
            num_node = G.num_node;
            max_degree = G.max_degree;
            pad_parameters();
            // tmp storage of sampled residuals
            std::vector<dist_t> sampled_residuals = std::vector<dist_t>(num_node * dimension, 0);
            std::vector<dist_t> sampled_query_residuals = std::vector<dist_t>(num_node * dimension, 0); 
            // rank for the system
            // tmp basis storage
            std::vector<std::vector<dist_t >> tmp_residual_basis;

            // sampled qres dres ip
            std::vector<dist_t > sampled_real_ip;
            std::vector<std::vector<dist_t >> residual1;
            std::vector<std::vector<dist_t >> residual2;
            // norm
            std::vector<dist_t> squared_norm_of_elements;

            // Stage : Build R-1 rank for each edge
            // collect sampled_residuals
            int total_valid_nodes = 0;
            int total_edge_links = 0;

            for (index_type i = 0; i < num_node; i++) {
                dist_t norm_ = do_dot_product_simd(
                    G.get_node_feat(i).val,
                    G.get_node_feat(i).val,
                    dimension
                );
                squared_norm_of_elements.push_back(norm_);
                const auto neighbors = G.get_neighborhood(i, 0);
                auto size = neighbors.degree();
                total_edge_links += size;
                if (size <= 1 || norm_ < 1e-6) {
                    continue;
                }

                total_valid_nodes += 1;
                std::uniform_int_distribution<> dis(0, size-1);
                int pick = dis(gen);
                int pick2 = dis(gen);

                while (pick == pick2) {
                    pick2 = dis(gen);
                    if (size == 1) {
                        break;
                    }
                }

               std::vector<dist_t > pick1_vec;
               std::vector<dist_t > pick2_vec;
              
               dist_t norm1 = 0;
               dist_t norm2 = 0;
               for (index_type j = 0; j < size; j++) {
                   G.prefetch_node_feat(neighbors[j + 1]);
                   auto next_node = neighbors[j];
                   if (j != pick & j != pick2) { 
                       continue;
                   }
                
                   dist_t dist = do_dot_product_simd(
                       G.get_node_feat(next_node).val,
                       G.get_node_feat(i).val,
                       dimension
                   );
                   dist_t* cc2 = G.get_node_feat(next_node).val;
                   dist_t* cc  = G.get_node_feat(i).val;
                   if (j == pick) {
                       for (int k = 0; k < dimension ; k++) {
                           sampled_residuals[i * dimension + k] =  cc2[k] -  dist / squared_norm_of_elements[i] * cc[k];
                           pick1_vec.push_back(sampled_residuals[i * dimension + k]);
                       }
                       for (int k = 0; k < dimension; k++) {
                          norm1 += ( pick1_vec[k] * pick1_vec[k] );
                       }
                                               
                       for(int k = 0; k < dimension; k++) { 
                           pick1_vec[k] = pick1_vec[k] / std::sqrt(norm1);}
                      }

                    if( j == pick2){
                          for (int k = 0; k < dimension ; k++) {
                              pick2_vec.push_back(  cc2[k] -  dist/squared_norm_of_elements[i] * cc[k] );
                              }
                          for(int k = 0; k < dimension; k++){
                              norm2 += ( pick2_vec[k] * pick2_vec[k] );
                               }
                          for(int k = 0; k < dimension; k++) { pick2_vec[k] = pick2_vec[k] / std::sqrt(norm2);}
                      }
               }
                if( norm1 != 0 && norm2 != 0){
                   if(pick != pick2)   {residual1.push_back(pick1_vec); }
                   if(pick != pick2) { residual2.push_back(pick2_vec);}
                   dist_t dist_ip = 0;
                   for(int k = 0 ; k < dimension; k++){ 
                       dist_ip += ( pick1_vec[k] * pick2_vec[k] );}
                   sampled_real_ip.push_back(dist_ip);
                }
          }
          // Calculate the Total Residual Basis
          Eigen::MatrixXf X(total_valid_nodes,dimension);
          for(int i = 0; i < total_valid_nodes; i++){
              for(int j = 0; j < dimension;j++){
                  X(i,j) = sampled_residuals[i * dimension + j];
              }
          }
          Eigen::BDCSVD<Eigen::MatrixXf> SVD(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
          Eigen::MatrixXf U = SVD.matrixU();
          Eigen::MatrixXf V = SVD.matrixV();
          //Eigen::MatrixXf recover2 = U * S * V.transpose();
          Eigen::MatrixXf projMatrix;
          float real_mean, appx_mean, real_std, appx_std;
          std::vector<dist_t > appx_ip;
          // Setup finger projection matrix
	  finger.projection_matrix.resize(low_rank * dimension, 0);
          for(int i = 0; i < low_rank; i++){
              for(int j = 0; j < dimension; j++){
                  finger.projection_matrix[i * dimension + j] = V(j,i);
              }
          } 

          // Low-dimensional center learning
          finger.low_rank = low_rank; 
          finger.dimension = dimension;
          //std::vector<float> low_residuals(total_edge_links * low_rank, 0);
          std::vector<index_type> encoded_result(total_edge_links, 0);
          std::vector<float> total_res_norm(total_edge_links, 0);
          std::vector<float> tmp_residual(dimension, 0);
          std::vector<float> tmp_low_residual(dimension, 0);

          int edge_cnt = 0;
          float dummy_a, dummy_b; 
/*
          for (size_t i = 0; i < num_node; i++) {
              const index_type size = *reinterpret_cast<const index_type*>(&G.buffer[G.mem_start_of_node[i]]);

              dist_t* center_node_feature  = G.get_node_feat(i).val;
                            
              float center_node_squared_norm = do_dot_product_simd(center_node_feature, center_node_feature, dimension); 
              for (index_type j = 0; j < size; j++) {
                  const index_type next_node = *reinterpret_cast<const index_type *>(&G.buffer[G.mem_start_of_node[i] + sizeof(index_type) + j * sizeof(index_type)]);
                  dist_t* neighbor_node_feature = G.get_node_feat(next_node).val;
                  dist_t dist = do_dot_product(center_node_feature, neighbor_node_feature, dimension);
                    
                  for (int k = 0; k < dimension ; k++) {
                      tmp_residual[k] = (neighbor_node_feature[k] -  dist / center_node_squared_norm * center_node_feature[k]);
                  }
                  total_res_norm[edge_cnt] = std::sqrt(do_dot_product_simd(tmp_residual.data(), tmp_residual.data(), dimension));
                  finger.compute_projection_information(tmp_residual.data(), &low_residuals[next_node * 32], dummy_a, dummy_b);
                  edge_cnt += 1; 
              }
          }
*/
            size_t neighbor_size = (1 + max_degree) * sizeof(index_type);
            code_offset = neighbor_size;
            node_mem_size = neighbor_size + 2 * sizeof(float) + low_rank * sizeof(float) + sizeof(uint64_t) * max_degree + max_degree * 2 * sizeof(float);   // node_only : center_node_norm : center_node_squared_norm : center_node_low_projection | neighbors : residual norm^2 ; center projection coefficient ; low-rank qres quantized index | neighbors : quantized vector index;
            
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;
 
            for (size_t i = 0; i < num_node; i++) {
                mem_start_of_node[i + 1] = mem_start_of_node[i] + node_mem_size;
            }
            buffer.resize(mem_start_of_node[num_node], 0);
            std::vector<float> neighbor_res_norm(max_degree, 0);
            //std::vector<float> neighbor_residual_center_negative_dot_product_result(max_degree, 0);
            std::vector<float> neighbor_center_projection_coefficient(max_degree, 0);
            //std::vector<index_type> neighbor_residual_quantized_idx(max_degree, 0);
            std::vector<uint64_t> neighbor_residual_codes(max_degree, 0);
/*
            for( int i = 0; i < 60000; i++) { 
                for( int j = 0; j < low_rank;j++) {
                std::cout<<finger.codebook[i * low_rank + j]<<",";} std::cout<<std::endl;
            }
*/
            std::vector<float> center_node_projection(low_rank, 0);
            edge_cnt = 0;
            for (size_t i = 0; i < num_node; i++) {
                memcpy(&buffer[mem_start_of_node[i]], &G.buffer[G.mem_start_of_node[i]], (1 + max_degree) * sizeof(index_type));
                const index_type size = *reinterpret_cast<const index_type*>(&G.buffer[G.mem_start_of_node[i]]);

                dist_t* center_node_feature  = G.get_node_feat(i).val;
                              
                float center_node_squared_norm = do_dot_product_simd(center_node_feature, center_node_feature, dimension); 
                float center_node_norm = std::sqrt(center_node_squared_norm);
                for (index_type j = 0; j < size; j++) {
                    const index_type next_node = *reinterpret_cast<const index_type *>(&G.buffer[G.mem_start_of_node[i] + sizeof(index_type) + j * sizeof(index_type)]);
                    dist_t* neighbor_node_feature = G.get_node_feat(next_node).val;
                    dist_t dist = do_dot_product(center_node_feature, neighbor_node_feature, dimension);

                    for (int k = 0; k < dimension ; k++) {
                        tmp_residual[k] =  (neighbor_node_feature[k] -  dist / center_node_squared_norm * center_node_feature[k]);
                    }

                    finger.compute_projection_information(tmp_residual.data(), tmp_low_residual.data(), dummy_a, dummy_b);
                    
                    neighbor_res_norm[j] = std::sqrt(do_dot_product_simd(tmp_residual.data(), tmp_residual.data(), dimension));
                    neighbor_center_projection_coefficient[j] = dist / center_node_squared_norm; 
                    //neighbor_res_norm[j] = total_res_norm[edge_cnt];
                    //neighbor_center_projection_coefficient[j] = dist / center_node_squared_norm;
                    uint64_t tmp = 0;
                    //if(j==0) { std::cout<<i<<"  "<<next_node<<" : ";}
                    
                    for (int r = 15; r >= 0; r--) {
                        if (tmp_low_residual[r] >= 0) {
                            tmp += 1;
                        }
                        tmp <<= 1;
                    }
                    for (int r = 31; r >= 16; r--) {
                        if (tmp_low_residual[r] >= 0) {
                            tmp += 1;
                        }
                        tmp <<= 1;
                    }
                    for (int r = 47; r >= 32; r--) {
                        if (tmp_low_residual[r] >= 0) {
                            tmp += 1;
                        }
                        tmp <<= 1;
                    }
                    for (int r = 63; r >= 48; r--) {
                        if (tmp_low_residual[r] >= 0) {
                            tmp += 1;
                        }
                        if ( r != 48) {
                            tmp <<= 1;
                        }
                    }

/*
                    for (int r = 0; r < 64; r++) { 
                        if (tmp_low_residual[r] >= 0) {
                            tmp += 1;
                            //if(j==0) { std::cout<<"1,";}
                        } else {
                            //if(j==0) { std::cout<<"0,";}
                        }                 
                        if (r != 63) {
                            tmp <<= 1;
                        }
                    }
*/
                    //if(j==0) { std::cout<<std::endl;}
                    //if(j==0) { for(int k = 0; k < 32; k++) {std::cout<<tmp_low_residual[k]<<",";} std::cout<<std::endl;}
                    neighbor_residual_codes[j] = tmp; 
                    //memcpy(&neighbor_residual_vectors[j * low_rank], &low_residuals[edge_cnt * low_rank], sizeof(float) * low_rank);
                    //neighbor_residual_quantized_idx[j] = finger.encode(tmp_low_residual.data());
                    //neighbor_residual_quantized_idx[j] = encoded_result[edge_cnt];
                    edge_cnt += 1; 
                }

                int num_groups = max_degree / 16;
                // save center node info 
                memcpy(&buffer[mem_start_of_node[i] + (1 + max_degree) * sizeof(index_type)], &center_node_norm, sizeof(float));
                memcpy(&buffer[mem_start_of_node[i] + (1 + max_degree) * sizeof(index_type) + 1 * sizeof(float)], &center_node_squared_norm, sizeof(float));
                size_t buffer_position = (1 + max_degree) * sizeof(index_type) + 2 * sizeof(float);
                // save center node low rank projection
                finger.compute_projection_information(center_node_feature, center_node_projection.data(), dummy_a, dummy_b);
                memcpy(&buffer[mem_start_of_node[i] + buffer_position], center_node_projection.data(), low_rank * sizeof(float));
                buffer_position += (low_rank * sizeof(float));                

                //memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_residual_quantized_idx[0], max_degree * sizeof(index_type));
                //memcpy(&buffer[mem_start_of_node[i] + buffer_position], neighbor_residual_codes.data(), max_degree * sizeof(uint32_t));
                //buffer_position += (max_degree * sizeof(uint32_t));
                
                // save neighboring node info 
                for (int j = 0; j < num_groups; j++) { 
                    // save to main buffer
                    memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_res_norm[j * 16], 16 * sizeof(float));
                    buffer_position += (16 * sizeof(float));
                    memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_center_projection_coefficient[j * 16], 16 * sizeof(float));
                    buffer_position += (16 * sizeof(float));
                    memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_residual_codes[j * 16], 16 * sizeof(uint64_t));
                    buffer_position += (16 * sizeof(uint64_t));
                    //memcpy(&buffer[mem_start_of_node[i] + buffer_position], neighbor_residual_vectors.data[j * 16 * low_rank], 16 * low_rank * sizeof(float));
                    //buffer_position += (16 * low_rank * sizeof(float));
                    //memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_residual_center_negative_dot_product_result[j * 16], 16 * sizeof(float));
                    //buffer_position += (16 * sizeof(float));
                    //memcpy(&buffer[mem_start_of_node[i] + buffer_position], &neighbor_residual_quantized_idx[j * 16], 16 * sizeof(index_type));
                    //buffer_position += (16 * sizeof(index_type));
                }
            }


          //finger.scale = (real_std / appx_std);
          //finger.bias =  (real_mean - appx_mean / appx_std * real_std);

        }

        inline const char* get_stored_info(index_type node_id) const {
            return &buffer[mem_start_of_node[node_id] + code_offset];
        }
        inline const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const {
            const index_type *neighborhood_ptr = nullptr;
            neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer.data()[node_id * (mem_index_type) node_mem_size]);
            return NeighborHood((void*)neighborhood_ptr);
        }
    };

