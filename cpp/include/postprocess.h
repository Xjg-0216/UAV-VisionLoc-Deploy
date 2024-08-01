#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif



#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "load_database.h"
#include <faiss/IndexFlat.h>
#include <vector>
#include <cmath> // exp pow
#include <numeric> //  std::iota  std::accumulate
#include <algorithm> //  std::sort
#include <utility> // std::pair

std::pair<float, float> post_process(faiss::IndexFlatL2& index, float* result, DatabaseData* db, int use_best_n);

#endif // __cplusplus

#endif // POST_PROCESS_H
