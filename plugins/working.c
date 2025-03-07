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

namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
}

::nlohmann::json GetJsonSpec(const char* path, const std::vector<int> shape, const std::vector<int> chunksize) {
  return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "file"},
             {"path", path},
         }},
        {"metadata",
         {
             {"zarr_format", 3},
             {"data_type", "float64"},
             {"shape", shape}
	}}};
}
::nlohmann::json GetJsonSpec2(const char* path, const int level, const std::vector<int> shape, const std::vector<int> chunksize) {
  return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "file"},
             {"path", path},
         }},
        {"metadata",
         {
             {"zarr_format", 3},
             {"data_type", "float64"},
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
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}

static auto CreateFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec;
	if(level==0)
	 json_spec = GetJsonSpec(path,shape,chunksize);
	else
	 json_spec = GetJsonSpec2(path,level,shape,chunksize);
	assert(json_spec==NULL);
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write);
return store_result;
}

static auto OpenFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
	::nlohmann::json json_spec = GetJsonSpec(path,shape,chunksize);
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write);	
return store_result;
}

//retieve 1 param data from driver
static auto createdata(json_object *main_obj, int i, std::vector<int> shape){
//const char *json_str3 = json_object_to_json_string(main_obj );
//std::cout <<"main obj:"<<json_str3;
json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
json_object *part_obj = json_object_array_get_idx(part_array, 0);
//std::cout <<"obj are fine:";
json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
//const char *json_str2 = json_object_to_json_string(vars_array );
//std::cout <<"vars are fine:";

json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
//struct json_object *data = json_object_array_get_idx(extarr_obj, 0);
//std::cout <<"data are fine:";
void const *buf = 0;
buf = json_object_extarr_data(extarr_obj);
const char *json_str1 = json_object_to_json_string(extarr_obj );
//std::cout <<"data is:"<<json_str1;

std::vector<double> parsed_data;
std::istringstream iss(json_str1);
char ch;
int value;
while (iss >> ch) {
	 if (std::isdigit(ch) || (ch == '-' && std::isdigit(iss.peek()))) {
                iss.putback(ch);
            iss >> value;
                parsed_data.push_back(static_cast<double>(value));
	     }
    }
//		    std::cout << "parsed_size: " << parsed_data.size() << std::endl;
std::cout <<"parsing are fine:";
std::vector<double> data_to_use(parsed_data.begin() + 4, parsed_data.end());
std::size_t size_data = data_to_use.size();
//std::cerr<<"size_data: "<<parsed_data[2];
//std::cerr<<" shape[0]: "<<shape[0];
//std::cerr<<" shape[1]: "<<shape[1];
//if (shape[0]>parsed_data[2]){
  //      shape[0]--;
    //    shape[1]--;
//std::cerr<<" reduce happend ";
//}

const Index rows = shape[0];
const Index cols = shape[1];
//const int *int_array = (const int *)buf;
//std::vector<double> int16_vec(vec.begin()+4, vec.end());
auto array=tensorstore::AllocateArray<double>({rows,cols});
//tensorstore::StridedLayout<2> array_layout(tensorstore::c_order, sizeof(double), {rows,cols});
//tensorstore::ArrayView<double, 2> array(&*(parsed_data.begin() + 4), array_layout);
//std::cout << "The size of the vector is: " << size_data << std::endl;
//std::cout << "rows: " << rows <<"cols" <<cols<<std::endl;
std::cout <<"alloc are fine:";
for (Index i = 0; i < rows; ++i) {
	for (Index j = 0; j < cols; ++j) {
		//std::cout << "writing: " << i*cols+j<<" i: "<<i<<" j: "<<j<<std::endl;
	    array(i,j) = data_to_use[i*cols+j];
	}
}
std::cout <<"define are fine:";
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
sprintf(fileName, "/u/asalimiparsa/apps/iometry2/zarr_%03d",dumpn);
auto shape= getshape(main_obj);
//std::cerr << "2 ";
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
 auto write_result = tensorstore::Write(data,create).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
timer_dt2 = MT_StopTimer(whole_timer);
//std::cerr << "5";
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
