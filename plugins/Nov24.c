#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

#include "tensorstore/internal/unowned_to_shared.h"
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
#include <tensorstore/driver/write.h>

namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::Array;
}

::nlohmann::json GetJsonSpec(const char* path, const std::vector<int> shape, const std::vector<int> chunksize,const int numthread) {
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
                    {"chunk_shape", chunksize}
                }}
			    }}}}};
}
::nlohmann::json GetJsonSpec2(const char* path, const int level, const std::vector<int> shape, const std::vector<int> chunksize,const int numthread) {
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
                {"codecs",
                        {
                                {{"name","gzip"},
                                        {"configuration",{{"level",level}}}}
                }}}
        }};
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
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}

static auto CreateFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec;
	if(level==0)
	 json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	else
	 json_spec = GetJsonSpec2(path,level,shape,chunksize,num_th);
	//assert(json_spec==NULL);
	::nlohmann::json json2=GetJsonSpec(path,shape,chunksize,num_th);
	auto store_result = tensorstore::Open(json2, context, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write);
return store_result;
}

static auto OpenFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
::nlohmann::json json_spec;
	json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write);	
return store_result;
}

//retieve 1 param data from driver
static auto createdata(json_object *main_obj, int i, std::vector<int> shape){
	json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
	json_object *part_obj = json_object_array_get_idx(part_array, 0);
	json_object *vars_array = json_object_path_get_array(part_obj, "Vars");

json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
json_object *int_obj = json_object_new_int(42);
json_object_put(int_obj);
int_obj = extarr_obj;
enum json_type type = json_object_get_type(int_obj);
switch(type){
	case json_type_null: std::cout <<"null";
	case json_type_double: std::cout << "Double";
	  case json_type_int:     std::cout << "Integer";
	   case json_type_object:  std::cout << "Object";
	 case json_type_array:  std::cout << "Array";
     case json_type_string:  std::cout << "String";				
	default:                std::cout << "Unknown";
	}
void const *buf = NULL;
buf = json_object_extarr_data(int_obj);
const char *json_str1 = json_object_to_json_string(extarr_obj);
const Index rows = shape[0];
const Index cols = shape[1];
//auto array_ref = tensorstore::Array(reinterpret_cast<const double*>(buf), {rows,cols}, tensorstore::c_order);
//auto shared_array = tensorstore::internal::UnownedToShared(array_ref);
const int32_t* double_buf = (const int32_t*)buf;
std::vector<double> parsed_data;
std::istringstream iss(json_str1);
char ch;
double value;
std::string number_str;
while (iss >> ch) {
        if (std::isdigit(ch) || ch == '-' || ch == '.' || ch == 'e' || ch == 'E') {        
	number_str.push_back(ch);}
	else if (!number_str.empty()) {
            std::istringstream num_stream(number_str);
            num_stream >> value;
            parsed_data.push_back(value);
            number_str.clear();
        }
    }
//std::cout <<"parsing are fine:";
std::vector<double> data_to_use(parsed_data.begin() + 4, parsed_data.end());
std::size_t size_data = parsed_data.size();
//std::cerr<<"size_data_parsed : "<<size_data;
//std::cerr<<" shape[0]: "<<shape[0];

if (shape[0]>parsed_data[2]){
        shape[0]--;
        shape[1]--;
//std::cerr<<" reduce happend ";
}

//std::cerr<<" shape[0]: "<<shape[0];
//std::cerr<<" shape[1]: "<<shape[1];
//std::cerr<<" rows: "<<rows;
//std::cerr<<" cols: "<<cols;
//const int *int_array = (const int *)buf;
auto array=tensorstore::AllocateArray<double>({rows,cols});
int32_t* d = const_cast<int32_t*>(double_buf);
//tensorstore::StridedLayout<2> array_layout(tensorstore::c_order, sizeof(int32_t), {rows,cols});
//auto array = tensorstore::MakeArray(&*(d), array_layout);
//tensorstore::ArrayView<int32_t, 2> array_view(&*(d), array_layout);
//std::cout << "The size of the vector is: " << size_data << std::endl;
//std::cout << "rows: " << rows <<"cols" <<cols<<std::endl;
//std::cout <<"alloc are fine:";
//auto source_array = UnownedToShared(
  //    Array(reinterpret_cast<double*>(buf), shape, c_order));

for (Index i = 0; i < rows; ++i) {
	for (Index j = 0; j < cols; ++j) {
	//	std::cout << "data: " << array_view(i,j)<<" i: "<<i<<" j: "<<j<<std::endl;
	    array(i,j) = parsed_data[i*cols+j];
	}
}

//std::cout <<"define are fine:";

return array;

//return 0;
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
//std::cerr << "1 ";
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
auto shape= getshape(main_obj);
//std::cerr << "2 ";
shape={16384,16384};
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata(main_obj,0,shape);
timer_dt = MT_StopTimer(main_dump_sif_tid);
//std::cerr << "3";
//timing for creating the zarr file
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
//timing for writing to zarr
//std::cerr << "4 ";
 main_dump_sif_tid = MT_StartTimer("Zarr_write_time", main_dump_sif_grp, dumpn);
 auto write_result = tensorstore::Write(tensorstore::UnownedToShared(data),create).status();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
timer_dt2 = MT_StopTimer(whole_timer);
//std::cerr << "5";
if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << std::endl;
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
