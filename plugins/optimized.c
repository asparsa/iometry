#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <random>

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
#include "tensorstore/util/endian.h"
std::mt19937 gen(42); // Use a fixed seed (42 in this case)
std::uniform_int_distribution<> distrib(0, 999);

namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
}
::nlohmann::json GetJsonSpec(const char* path, const std::vector<int> shape, const std::vector<int> chunksize, const int numthread, std::optional<int> level = std::nullopt) {
    ::nlohmann::json json_spec = {
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "file"},
             {"path", path},
             {"file_io_concurrency", {{"limit", numthread}}}
         }},
        {"metadata",
         {
             {"zarr_format", 3},
             {"data_type", "float64"},
             {"shape", shape},
             {"chunk_grid", {
                 {"name", "regular"},
                 {"configuration", {
                     {"chunk_shape", chunksize}
                 }}
             }}
         }}
    };

    if (level.has_value()) {
        json_spec["metadata"]["codecs"] = {
            {
                {"name", "gzip"},
                {"configuration", {{"level", level.value()}}}
            }
        };
    }

    return json_spec;
}

::nlohmann::json GetJsonSpec2(const char* path, const std::vector<int> shape, const std::vector<int> chunksize,const int numthread) {
  return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "file"},
             {"path", path},
	     {"file_io_concurrency",{{"limit",numthread}}}
         }},
        {"metadata",
         {
             {"zarr_format", 3},
             {"data_type", "int32"},
             {"shape", shape},
	     {"chunk_grid", {
                {"name", "regular"},
                {"configuration", {
                    {"chunk_shape", shape}
                }}
			    }}}}};
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
static int dim2=0;
static int level=0;
static int num_th=1;
static int shape0=0;
static int shape1=0;
static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
	    "--compression_level %d","",
	    "level of compression with gzip",
	    &level,
	    "--chunk_size %d %d","",
	    "size of the chunks",
	    &dim1,&dim2,
	    "--num_threads %d","",
	    "number of threads for writing",
	    &num_th,
	    "--dims %d %d","",
	    "dimension of data",
	    &shape0,&shape1,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}
//create tensorstore file
static auto CreateFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec=GetJsonSpec2(path,shape,chunksize,num_th);
	/*
	if(level==0)
	 json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	else
	 json_spec = GetJsonSpec(path,shape,chunksize,num_th,level);
	*/
	 assert(json_spec.empty());
	
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write);
return store_result;
}
//open the tensorstore, not applicant yet
static auto OpenFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
::nlohmann::json json_spec;
	json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write);	
return store_result;
}

std::vector<double> copy_array(double *source, unsigned long long length) {
	std::vector<double> destination(length);
    std::memcpy(destination.data(), source, length * sizeof(double));
    return destination;
}
/*
//create zarr file 
static auto createdata2(json_object *main_obj, int i, std::vector<int> shape){
//this lines gets the data from json and save it to a array	
json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
json_object *part_obj = json_object_array_get_idx(part_array, 1);
json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
double *valdp;
valdp = (double *) json_object_extarr_data(extarr_obj);
unsigned long long leng=shape[0]*shape[1];
std::vector<double> vec(valdp, valdp + leng);
auto array2=tensorstore::Array(vec, {shape[0]-1,shape[1]-1}, tensorstore::c_order);
std::vector<double> result =copy_array(valdp, leng);
const Index rows[2] = {shape[0]-1,shape[1]-1};
const Index cols = shape[1]-1;
//auto array = tensorstore::MakeArray<double>(result);
return array2;
}*/
//create data directly

static auto createdata2(json_object *main_obj, int type){
const Index rows = shape0;
const Index cols = shape1;
auto array=tensorstore::AllocateArray<double>({rows,cols});

if (type==0){
for (Index i=0; i<rows;i++)
	for(Index j=0;j<cols;j++)
		array(i,j)=(double) distrib(gen) / 1000;
}


return array;
}

//get the shape from data created by driver
static auto getshape(json_object *main_obj){
std::vector<int> shape;
json_object *global_log_dims_array = json_object_path_get_array(main_obj, "problem/global/LogDims");
 int array_len = json_object_array_length(global_log_dims_array);
for (int i = 0; i < array_len; ++i) {
            struct json_object *element_obj = json_object_array_get_idx(global_log_dims_array, i);
            int element_value = json_object_get_int(element_obj);
            shape.push_back(element_value-1);
        }
return shape;
}

static void main_dump(int argi, int argc, char **argv, json_object *main_obj, int dumpn, double dumpt) {
//timings
MACSIO_TIMING_GroupMask_t main_dump_sif_grp = MACSIO_TIMING_GroupMask("main_dump_sif");
MACSIO_TIMING_TimerId_t main_dump_sif_tid;
MACSIO_TIMING_TimerId_t whole_timer;
double timer_dt;
double timer_dt2;

process_args(argi, argc, argv);
whole_timer=MT_StartTimer("whole_time", main_dump_sif_grp, dumpn);

//creating the metadata
char fileName[256];
std::vector<int> chunksize={dim1,dim2};
sprintf(fileName, "zarr_%03d",dumpn);
//auto shape= getshape(main_obj);
std::vector<int> shape={shape0,shape1};
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid);
//time to create zarr array
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata2(main_obj,0);
timer_dt = MT_StopTimer(main_dump_sif_tid);
/*
//timing for creating the zarr file
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
*/
//timing for writing to zarr
 main_dump_sif_tid = MT_StartTimer("Zarr_write_time", main_dump_sif_grp, dumpn);
 auto write_result = tensorstore::Write(data,create).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
timer_dt2 = MT_StopTimer(whole_timer);
/*
if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << write_result.status() << std::endl;
}
else std::cerr << "succesful write "; 
*/
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
