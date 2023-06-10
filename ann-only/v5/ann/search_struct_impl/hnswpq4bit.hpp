    template<typename dist_t, class FeatVec_T>
    struct HNSWProductQuantizer4Bits {
        typedef FeatVec_T feat_vec_t;
        typedef Pair<dist_t, index_type> pair_t;
        typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
        typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;


        // scalar variables
        index_type num_node;
        index_type maxM;   // max number of out-degree for level l=1,...,L
        index_type maxM0;  // max number of out-degree for level l=0
        index_type efC;    // size of priority queue for construction time
        index_type max_level;
        index_type init_node;
        index_type subspace_dimension;  // dimension of each subspace in Product Quantization
        index_type sub_sample_points;   // number of sub-sampled points used to build quantizer subspace centors. 

        GraphL0<feat_vec_t> feature_vec;           // feature vectors only
        GraphL1 graph_l1;                       // neighborhood graphs from level 1 and above
        GraphProductQuantizer4Bits<feat_vec_t> graph_l0_pq4;   // Productquantized4Bits neighborhood graph built from graph_l0
        HNSWProductQuantizer4Bits() {
            std::string space_type = pecos::type_util::full_name<feat_vec_t>();
            if (space_type != "pecos::ann::FeatVecDenseL2Simd<float>") {
                throw std::runtime_error("Currently, we only support L2 distance with float type.");
            } 
        }
        ~HNSWProductQuantizer4Bits() {}
        struct Searcher : SetOfVistedNodes<unsigned short int> {
            typedef SetOfVistedNodes<unsigned short int> set_of_visited_nodes_t;
            typedef HNSWProductQuantizer4Bits<dist_t, FeatVec_T> hnswpq4_t;
            typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
            typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;

            const hnswpq4_t* hnsw;
            max_heap_t topk_queue;
            min_heap_t cand_queue;
            alignas(64) std::vector<uint8_t> lut;
            alignas(64) std::vector<float> appx_dist;
            float scale;
            float bias;

            Searcher(const hnswpq4_t* _hnsw=nullptr):
                SetOfVistedNodes<unsigned short int>(_hnsw? _hnsw->num_node : 0),
                hnsw(_hnsw)
            {}

            void reset() {
                set_of_visited_nodes_t::reset();
                topk_queue.clear();
                cand_queue.clear();
            }

            void prepare_inference() {
                auto num_of_local_centroids = hnsw->graph_l0_pq4.quantizer.num_of_local_centroids;
                auto max_degree = hnsw->graph_l0_pq4.max_degree;
                auto num_local_codebooks = hnsw->graph_l0_pq4.quantizer.num_local_codebooks;

                //  When using AVX512f, we have 16 centroids per local codebook, and each of it uses 8 bits to represent quantized
                //  distance value. Thus,m we will have 128 bits to load 1 set of local codebooks. Thus, a loadu_si512 will load
                //  512 / 128 == 4 local codebooks at a time. Thus, the lookup table size needs to be adjusted (padding 0) if
                //  if num_local_codebooks is not divisible by 4.
                size_t adjusted_num_local_codebooks = num_local_codebooks % 4 == 0 ? num_local_codebooks : (num_local_codebooks / 4 + 1) * 4;

                // Similarly, we have to parse every 16 neighbors at a time to maximally leverage avx512f.
                // Thus, we have to prepare result array which is multiple of 16 to make sure the SIMD
                // will not touch unavailable memory
                size_t adjusted_max_degree = max_degree % 16 == 0 ? max_degree : ((max_degree / 16) + 1) * 16;

                lut.resize(num_of_local_centroids * adjusted_num_local_codebooks, 0);
                appx_dist.resize(adjusted_max_degree, 0);
            }
            void setup_lut(float* query) {
                hnsw->graph_l0_pq4.quantizer.setup_lut(query, lut.data(), scale, bias);
            }
            void approximate_distance(size_t neighbor_size, const char* neighbor_codes) {
                // pass searcher to group_distance
                hnsw->graph_l0_pq4.quantizer.approximate_neighbor_group_distance(neighbor_size, appx_dist.data(), neighbor_codes, lut.data(), scale, bias);
            }

            max_heap_t& search_level(const feat_vec_t& query, index_type init_node, index_type efS, index_type level) {
                return hnsw->search_level(query, init_node, efS, level, *this);
            }

            max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, index_type num_rerank) {
                return hnsw->predict_single(query, efS, topk, *this, num_rerank);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }


        static nlohmann::json load_config(const std::string& filepath) {
            std::ifstream loadfile(filepath);
            std::string json_str;
            if (loadfile.is_open()) {
                json_str.assign(
                    std::istreambuf_iterator<char>(loadfile),
                    std::istreambuf_iterator<char>());
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_param = nlohmann::json::parse(json_str);
            std::string hnsw_t_cur = pecos::type_util::full_name<HNSWProductQuantizer4Bits>();
            std::string hnsw_t_inp = j_param["hnsw_t"];
            if (hnsw_t_cur != hnsw_t_inp) {
                throw std::invalid_argument("Inconsistent HNSW_T: hnsw_t_cur = " + hnsw_t_cur  + " hnsw_t_cur = " + hnsw_t_inp);
            }
            return j_param;
        }

        void save_config(const std::string& filepath) const {
            nlohmann::json j_params = {
                {"hnsw_t", pecos::type_util::full_name<HNSWProductQuantizer4Bits>()},
                {"version", "v1.0"},
                {"train_params", {
                    {"num_node", this->num_node},
                    {"subspace_dimension", this->subspace_dimension},
                    {"sub_sample_points", this->sub_sample_points},
                    {"maxM", this->maxM},
                    {"maxM0", this->maxM0},
                    {"efC", this->efC},
                    {"max_level", this->max_level},
                    {"init_node", this->init_node}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if (savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void save(const std::string& model_dir) const {
            if (mkdir(model_dir.c_str(), 0777) == -1) {
                if (errno != EEXIST) {
                    throw std::runtime_error("Unable to create save folder at " + model_dir);
                }
            }
            save_config(model_dir + "/config.json");
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "wb");
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM0, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&efC, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&init_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&subspace_dimension, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&sub_sample_points, 1, fp);
            feature_vec.save(fp);
            graph_l1.save(fp);
            graph_l0_pq4.save(fp);
            fclose(fp);
        }

        void load(const std::string& model_dir) {
            auto config = load_config(model_dir + "/config.json");
            std::string version = config.find("version") != config.end() ? config["version"] : "not found";
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "rb");
            if (version == "v1.0") {
                pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM0, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&efC, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&init_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&subspace_dimension, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&sub_sample_points, 1, fp);
                feature_vec.load(fp);
                graph_l1.load(fp);
                graph_l0_pq4.load(fp);
            } else {
                throw std::runtime_error("Unable to load this binary with version = " + version);
            }
            fclose(fp);
        }

        template<class MAT_T>
        void train(
            const MAT_T &X_trn,
            index_type M,
            index_type efC,
            index_type subspace_dimension=0,
            index_type sub_sample_points=0,
            int threads=1,
            int max_level_upper_bound=-1
        ) {
            HNSW<dist_t, feat_vec_t>* hnsw = new HNSW<dist_t, feat_vec_t>();
            hnsw->train(X_trn, M, efC, threads, max_level_upper_bound);
            this->num_node = hnsw->num_node;
            this->maxM = hnsw->maxM;
            this->maxM0 = hnsw->maxM0;
            this->efC = hnsw->efC;
            this->max_level = hnsw->max_level;
            this->init_node = hnsw->init_node;
            this->subspace_dimension = subspace_dimension;
            this->sub_sample_points = sub_sample_points;

            graph_l1.num_node = hnsw->graph_l1.num_node;
            graph_l1.max_level = hnsw->graph_l1.max_level;
            graph_l1.max_degree = hnsw->graph_l1.max_degree;
            graph_l1.node_mem_size = hnsw->graph_l1.node_mem_size;
            graph_l1.level_mem_size = hnsw->graph_l1.level_mem_size;
            graph_l1.buffer.resize(hnsw->graph_l1.buffer.size());
            memcpy(graph_l1.buffer.data(), hnsw->graph_l1.buffer.data(), hnsw->graph_l1.buffer.size() * sizeof(index_type));

            graph_l0_pq4.build_quantizer(X_trn, subspace_dimension, sub_sample_points);
            graph_l0_pq4.build_graph(hnsw->graph_l0);
            delete hnsw;
            feature_vec.init(X_trn, -1);
        }


        max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, Searcher& searcher, index_type num_rerank) const {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = feature_vec;
            // specialized search_level for level l=1,...,L because its faster for efS=1
            dist_t curr_dist = feat_vec_t::distance(
                query,
                G0.get_node_feat(init_node)
            );
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto neighbors = G1.get_neighborhood(curr_node, curr_level);
                    if (neighbors.degree() != 0) {
                        feature_vec.prefetch_node_feat(neighbors[0]);
                        index_type max_j = neighbors.degree() - 1;
                        for (index_type j = 0; j <= max_j; j++) {
                            feature_vec.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
                            auto next_node = neighbors[j];
                            dist_t next_dist = feat_vec_t::distance(
                                query,
                                G0.get_node_feat(next_node)
                            );
                            if (next_dist < curr_dist) {
                                curr_dist = next_dist;
                                curr_node = next_node;
                                changed = true;
                            }
                        }
                    }
                }
            }
            // generalized search_level for level=0 for efS >= 1
            searcher.search_level(query, curr_node, std::max(efS, topk), 0);
            auto& topk_queue = searcher.topk_queue;


            if (num_rerank > 0) {
                index_type t_size = topk_queue.size() > num_rerank ? topk_queue.size() - num_rerank : 0;
                for (index_type i = 0; i < t_size; i++) {
                    topk_queue.pop();
                }
                for (auto i = topk_queue.begin(); i != topk_queue.end(); ++i) {
                    feature_vec.prefetch_node_feat((*(i + 1)).node_id);
                    pair_t cand_pair = (*i);
                    dist_t next_dist = feat_vec_t::distance(
                        query,
                        G0.get_node_feat(cand_pair.node_id)
                    );
                    (*i).dist = next_dist;
                }
                std::sort(topk_queue.begin(), topk_queue.end());
                if (topk_queue.size() > topk) {
                    topk_queue.resize(topk);
                }
                return searcher.topk_queue;
            }



            if (topk < efS) {
                // remove extra when efS > topk
                while (topk_queue.size() > topk) {
                    topk_queue.pop();
                }
            }
            std::sort_heap(topk_queue.begin(), topk_queue.end());
            return topk_queue;
        }

        max_heap_t& search_level(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            const auto *G0Q = &graph_l0_pq4;
            searcher.reset();
            searcher.setup_lut(query.val);
            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                feature_vec.get_node_feat(init_node)
            );
            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
            searcher.mark_visited(init_node);

            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;

                // visiting neighbors of candidate node
                const auto neighbors = G0Q->get_neighborhood(cand_node, level);
                if (neighbors.degree() != 0) {
                    index_type max_j = neighbors.degree() - 1;

                    searcher.approximate_distance(max_j + 1, G0Q->get_neighbor_codes(cand_node));
                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        dist_t next_lb_dist = searcher.appx_dist[j];

                        if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                            if (!searcher.is_visited(next_node)) {
                                searcher.mark_visited(next_node);
                                cand_queue.emplace(next_lb_dist, next_node);
                                topk_queue.emplace(next_lb_dist, next_node);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }
                }

            }



            return topk_queue;
        }
    };
