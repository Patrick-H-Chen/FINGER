            for (int iter = 0; iter < max_iter; iter++) {
                std::cout<<"Clustering Iteration : "<<iter<<std::endl;
                int repeat_times = n_data / threads; 
                for (int r = 0; r < repeat_times; r++) {
                    #pragma omp parallel num_threads(threads)
                    {
                        int rank = omp_get_thread_num();
                        encoded_result[r * threads + rank] = encode(&X_trn[(r * threads + rank) * low_rank]); //do_l2_distance_simd(vector, &codebook[(r * threads + rank) * low_rank], low_rank);
                    }
                }
/*
std::cout<<"AFTER ENCODING : "<<std::endl;

            for (int i = 0; i < num_codebooks; i++) {
                for (int j = 0; j < low_rank; j++) {
                    std::cout<<codebook[i * low_rank + j]<<",";
                }
                std::cout<<std::endl;
            }
*/
                for (int i = repeat_times * threads; i < n_data; i++) {
                    encoded_result[i] = encode(&X_trn[i * low_rank]);
                }

                repeat_times = num_codebooks / threads;
                int repeat_dimension = low_rank / 16;
                for (int r = 0; r < repeat_times; r++) {
                    #pragma omp parallel num_threads(threads)
                    {
                        int rank = omp_get_thread_num();
                        int codebook_num = r * threads + rank;
                        int count = 0;

                       // std::vector<__m512> tmp_center(repeat_dimension);
                       // for (int k = 0; k < repeat_dimension; k++) { 
                        //    tmp_center[k] = _mm512_setzero_ps();
                       // }
                       // std::cout<<codebook_num<<std::endl;

                        std::vector<float > tmp_center(low_rank, 0);
                        for (int i = 0; i < n_data; i++) {
                            if (encoded_result[i] == codebook_num) {
                                for (int r = 0; r < low_rank; r++) { 
                                    tmp_center[r] += X_trn[i * low_rank + r];
                                }
                                //for (int k = 0; k < repeat_dimension; k++) {
                                //    __m512 _x = _mm512_loadu_ps(&X_trn[i * low_rank + k * 16]); 
                                //    tmp_center[k] = _mm512_add_ps(tmp_center[k], _x);
                                //} 
                                count += 1; 
