

#include <cstdio>  
#include <cstdlib> 


#ifndef UTILS_H_
#define UTILS_H_


#ifdef __cplusplus
extern "C" {
#endif

int read_data_from_file(const char *path, char **out_data);
#ifdef __cplusplus
}
#endif

#endif // UTILS_H_