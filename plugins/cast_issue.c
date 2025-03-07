#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <json-cwx/json.h>
#include <macsio_clargs.h>
#include <macsio_iface.h>
#include <macsio_log.h>
#include <macsio_main.h>
#include <macsio_mif.h>
#include <macsio_timing.h>
#include <macsio_utils.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/tensorstore.h"
#include <tensorstore/index_space/dim_expression.h>
#include "tensorstore/util/result.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"


namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
}
::nlohmann::json GetJsonSpec(const char* path, const std::vector<int> shape, const std::vector<int> chunksize) {
  return {
{"driver", "zarr3"},
      {"kvstore", {{"driver", "file"}, {"path", "/tmp/macsio"}, {"file_io_sync",true}}},
      {"metadata",
       {
           {"data_type", "int16"},
           {"shape", shape},
           {"chunk_grid", {
                {"name", "regular"},
                {"configuration", {
                    {"chunk_shape", shape}
                }}
       }}
  }}
};
  }
static auto context = Context::Default();
static char const *iface_name = "zarr";
static char const *iface_ext = "zarr";

static int use_log = 0;
static int no_collective = 0;
static int no_single_chunk = 0;
static const char *filename;
static tensorstore::TensorStore<> store;
static int show_errors = 0;
static int dim1=0;

static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}

static auto CreateFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec = GetJsonSpec(path,shape,chunksize);
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write);
return store_result;
}

static auto OpenFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec = GetJsonSpec(path,shape,chunksize);
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write);	
return store_result;
}
void copy_to_vector(void const *buf, size_t buf_size, std::vector<int> &vec) {
    // Ensure the buffer size is a multiple of sizeof(int)
    if (buf_size % sizeof(int) != 0) {
        std::cerr << "Buffer size is not a multiple of sizeof(int)" << std::endl;
        return;
    }

    // Calculate the number of integers in the buffer
    size_t num_elements = buf_size / sizeof(int);

    // Resize the vector to hold the data
    vec.resize(num_elements);

    // Copy the data from the buffer to the vector
    std::memcpy(vec.data(), buf, buf_size);
}

//retieve 1 param data from driver
static auto createdata(json_object *main_obj, int i, std::vector<int> shape){
json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
json_object *part_obj = json_object_array_get_idx(part_array, 0);
json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
//struct json_object *data = json_object_array_get_idx(extarr_obj, 0);
void const *buf = 0;
buf = json_object_extarr_data(extarr_obj);
unsigned char* raw_buf = static_cast<unsigned char*>(const_cast<void*>(buf));

//const char *json_str1 = json_object_to_json_string(extarr_obj );
const Index rows = shape[0];
const Index cols = shape[1];
size_t buf_size=rows*cols+4;
//const int *int_array = (const int *)buf;
//std::vector<int16_t> int16_vec(vec.begin()+4, vec.end());
for(int i=0;i<40;i++) std::cout<<static_cast<int>(raw_buf[i]) << " ";
auto array = tensorstore::MakeArray({1,2});
/*
for (Index i = 0; i < rows-1; ++i) {
	for (Index j = 0; i < cols-1; ++j) {
            struct json_object *element_data = json_object_array_get_idx(extarr_obj, (rows*i+j));
	if (element_data!=NULL)
	    array(i,j) = json_object_get_int(element_data);
    	else break;
	}
}
*/
return array;
}
static void main_dump(int argi, int argc, char **argv, json_object *main_obj, int dumpn, double dumpt) {
//timings
MACSIO_TIMING_GroupMask_t main_dump_sif_grp = MACSIO_TIMING_GroupMask("main_dump_sif");
MACSIO_TIMING_TimerId_t main_dump_sif_tid;
double timer_dt;
//creating the metadata
char fileName[256];
sprintf(fileName, "zarr_%03d",dumpn);
std::vector<int> shape;
json_object *global_log_dims_array = json_object_path_get_array(main_obj, "problem/global/LogDims");
 int array_len = json_object_array_length(global_log_dims_array);
for (int i = 0; i < array_len; ++i) {
            struct json_object *element_obj = json_object_array_get_idx(global_log_dims_array, i);
            int element_value = json_object_get_int(element_obj);
            shape.push_back(element_value);
        }
main_dump_sif_tid = MT_StartTimer("create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
auto data=createdata(main_obj,0,shape);
 main_dump_sif_tid = MT_StartTimer("write_time", main_dump_sif_grp, dumpn);
auto write_result = tensorstore::Write(data,create).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 

  if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << write_result.status() << std::endl;
}
else std::cerr << "succesful write "; 
}
static int register_this_interface() {
    MACSIO_IFACE_Handle_t iface;

    if (strlen(iface_name) >= MACSIO_IFACE_MAX_NAME)
        MACSIO_LOG_MSG(Die, ("Interface name \"%s\" too long", iface_name));

    strcpy(iface.name, iface_name);
    strcpy(iface.ext, iface_ext);
    iface.dumpFunc = main_dump;
    iface.processArgsFunc = process_args;

    if (!MACSIO_IFACE_Register(&iface))
        MACSIO_LOG_MSG(Die, ("Failed to register interface \"%s\"", iface_name));

    return 0;
}

static int dummy = register_this_interface();
