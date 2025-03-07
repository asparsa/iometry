#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <random>
#include <fstream>  // For file handling
#include <iostream> 

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
#include "tensorstore/data_type.h"
namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using tensorstore::dtypes::int64_t;
using tensorstore::dtypes::float16_t;
}
static int num_th=1;
static int dim1=1;
static int dim2=1;
::nlohmann::json readjson(const char* path) {
  return {
      {"driver", "zarr3"},
      {"kvstore", {{"driver", "file"}, {"path", path}}},
      {"metadata",
       {{"data_type", "int32"},
       }}};
}
::nlohmann::json writejson(int th, std::vector<int> shape,const char* path, std::vector<int> chunksize) {
  return {
      {"driver", "zarr3"},
      {"kvstore", {
          {"driver", "file"},
          {"path", path}
      }},
      {"metadata", {
          {"data_type", "int32"},
          {"shape", shape},
	  {"chunk_grid",
          {{"name","regular"},
           {"configuration", {{"chunk_shape", chunksize}}}
             } }}}
  };
}
::nlohmann::json writejson2(int th, std::vector<int> shape,const char* path, std::vector<int> chunksize) {
    return {
        {"driver", "zarr"},
        {"kvstore", {
            {"driver", "file"},
            {"path", path}
        }},
        {"metadata", {
            {"dtype", "<i4"},
            {"shape", shape},
            {"chunks", chunksize}}}
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
static int level=0;
static int shape2=1;
static int shape1=1;
static char read_type[64];
static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
	 	"--read_type %s", "",
	    "choose between FULL, RANDOM, hyperslap, overlaping hyperslap",
		&read_type,
	    "--compression_level %d","",
	    "level of compression with gzip",
	    &level,
	    "--chunk_size %d %d",MACSIO_CLARGS_NODEFAULT,
	    "size of the chunks",
	    &shape1,&shape2,
	    "--num_threads %d","",
	    "number of threads for writing",
	    &num_th,
	    "--dims %d %d",MACSIO_CLARGS_NODEFAULT,
	    "dimension of data",
	    &dim1,&dim2,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}
//create tensorstore file
static auto CreateFile( std::vector<int> shape, const char* path,std::vector<int> chunksize ){
	//::nlohmann::json json_spec=GetJsonSpec2(path,shape,chunksize,num_th);
	/*
	if(level==0)
	 json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	else
	 json_spec = GetJsonSpec(path,shape,chunksize,num_th,level);
	*/
	 //assert(json_spec.empty());
//printf("the num_th is=%d", num_th);	
auto store_result = tensorstore::Open(writejson(num_th,shape,path,chunksize), context, tensorstore::OpenMode::create, tensorstore::ReadWriteMode::read_write);
return store_result;
}
//open the tensorstore, not applicant yet
/*
static auto OpenFile( const char* path,const std::vector<int> shape, const std::vector<int> chunksize){
::nlohmann::json json_spec;
	json_spec = GetJsonSpec(path,shape,chunksize,num_th);
	auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write);	
return store_result;
}
*/
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

static auto createdata2(json_object *main_obj, int type,std::vector<int> shape){
const Index rows = shape[0];
const Index cols = shape[1];
const Index z=shape[1];
auto array=tensorstore::AllocateArray<int>({rows,cols});

if (type==0){
for (Index i=0; i<rows;i++)
	for(Index j=0;j<cols;j++)
	//for(Index q=0;q<z;q++)
		array(i,j)=1;
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

static void main_dump(int argi, int argc, char **argv, json_object *main_obj, int dumpn, double dumpt, const char* writepath) {
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
std::vector<int> chunksize={shape1,shape2};
//auto shape= getshape(main_obj);
std::vector<int> shape={dim1,dim2};
//printf("dim1=%d dim2=%d",dim1,dim2);
//time to create zarr array
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata2(main_obj,0,shape);
timer_dt = MT_StopTimer(main_dump_sif_tid);
/*
//timing for creating the zarr file
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
*/
//for(int i=0;i<1000;i++){
sprintf(fileName, "%szarr_%03d",writepath,dumpn);
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(shape,fileName,chunksize).result();
timer_dt = MT_StopTimer(main_dump_sif_tid);
//timing for writing to zarr
 main_dump_sif_tid = MT_StartTimer("Zarr_write_time", main_dump_sif_grp, dumpn);
 auto write_result = tensorstore::Write(data,create).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
timer_dt2 = MT_StopTimer(whole_timer);

if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << write_result.status() << std::endl;
}
else std::cerr << "succesful write "; 
}

static void main_read_full(const char* path,json_object *main_obj, int loadnumber )
{
	MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_full");
        MACSIO_TIMING_TimerId_t main_read_full_tid;
        double timer_dt;
	
	::nlohmann::json json_spec = readjson(path);
	std::ofstream file("metadata.json");
	file << json_spec.dump(4);  // The argument '4' makes the JSON output pretty
    file.close();
	main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open);
        timer_dt = MT_StopTimer(main_read_full_tid);
	int16_t* data = (int16_t*)malloc(16000 * 16000 * sizeof(int16_t));
	auto store = std::move(result).value();
	main_read_full_tid = MT_StartTimer("read_data", main_read_full_grp, loadnumber);
	auto read_result = tensorstore::Read(
        store).result();
	timer_dt = MT_StopTimer(main_read_full_tid);
	if (!read_result.ok()) {
        std::cerr << "Failed to read data: " << read_result.status() << std::endl;
        free(data);
    }
	//std::cerr <<read_result.value();
	free(data);
}

static void main_read_hyper(const char* path,json_object *main_obj, int loadnumber ){
	MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_random");
         MACSIO_TIMING_TimerId_t main_read_full_tid;
         double timer_dt;
	std::cerr << "reading hyper ";
	::nlohmann::json json_spec = readjson(path);
	main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
	auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
	timer_dt = MT_StopTimer(main_read_full_tid);
	std::vector<std::array<int, 2>> indices = {{3, 5}, {8, 2}, {1, 4}};
	int16_t* data = (int16_t*)malloc(8000 * 8000 * sizeof(int16_t));
        //auto randomread = tensorstore::Read(
          //          store , tensorstore::AllDims().TranslateSizedInterval(
            //                    {9, 7}, {1, 1}));
        main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
	auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 1000}, {1600, 1600}))
                    .result();
	timer_dt = MT_StopTimer(main_read_full_tid);
        main_read_full_tid = MT_StartTimer("read_data_huge", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result2=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 1000}, {8000, 8000}))
                    .result();
        timer_dt = MT_StopTimer(main_read_full_tid);
	if (!result.ok()) {
        std::cerr << "Failed to read data: " << result.status() << std::endl;
        free(data);
    }
	   
        free(data);
	
	
}

static void main_read_multihyper(const char* path,json_object *main_obj, int loadnumber ){
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_random");
         MACSIO_TIMING_TimerId_t main_read_full_tid;
         double timer_dt;
        std::cerr << "reading hyper ";
        ::nlohmann::json json_spec = readjson(path);
        main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
        timer_dt = MT_StopTimer(main_read_full_tid);
        std::vector<std::array<int, 2>> indices = {{3, 5}, {8, 2}, {1, 4}};
        int16_t* data = (int16_t*)malloc(8000 * 8000 * sizeof(int16_t));
        //auto randomread = tensorstore::Read(
          //          store , tensorstore::AllDims().TranslateSizedInterval(
            //                    {9, 7}, {1, 1}));
        main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 1000}, {800, 800}))
                    .result();
	auto result3=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {2000, 2000}, {800, 800}))
                    .result();
auto result4=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {3000, 3000}, {800, 800}))
                    .result();
auto result5=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {4000, 4000}, {800, 800}))
                    .result();
        timer_dt = MT_StopTimer(main_read_full_tid);
        main_read_full_tid = MT_StartTimer("read_data_huge", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result2=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {5000, 5000}, {2000, 2000}))
                    .result();
        auto result13=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {7000, 7000}, {2000, 2000}))
                    .result();
auto result12=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {9000, 9000}, {2000, 2000}))
                    .result();
auto result11=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {11000, 11000}, {2000, 2000}))
                    .result();
	timer_dt = MT_StopTimer(main_read_full_tid);
        if (!result.ok()) {
        std::cerr << "Failed to read data: " << result.status() << std::endl;
        free(data);
    }   
    
        free(data);
    
    
}


static void main_read(int argi, int argc, char **argv, const char* path, json_object *main_obj,int loadnumber){
	MACSIO_TIMING_GroupMask_t main_read_grp = MACSIO_TIMING_GroupMask("main_read");
	MACSIO_TIMING_TimerId_t main_read_tid;
    	double timer_dt;
	process_args(argi, argc, argv);
	if(!strcmp(read_type,"FULL")){
	main_read_tid=MT_StartTimer("main_read_FULL",main_read_grp,loadnumber);
	main_read_full(path, main_obj,loadnumber);
       	timer_dt= MT_StopTimer(main_read_tid);
	}
	else if(!strcmp(read_type,"HYPER")){
		main_read_tid=MT_StartTimer("main_read_FULL",main_read_grp,loadnumber);
		main_read_hyper(path, main_obj,loadnumber);
		timer_dt= MT_StopTimer(main_read_tid);
	}
	else{
                main_read_tid=MT_StartTimer("main_read_FULL",main_read_grp,loadnumber);
                main_read_multihyper(path, main_obj,loadnumber);
                timer_dt= MT_StopTimer(main_read_tid);
        }	

}
static int register_this_interface() {
    MACSIO_IFACE_Handle_t iface;

    if (strlen(iface_name) >= MACSIO_IFACE_MAX_NAME)
        MACSIO_LOG_MSG(Die, ("Interface name \"%s\" too long", iface_name));

    strcpy(iface.name, iface_name);
    strcpy(iface.ext, iface_ext);
    iface.dumpFunc = main_dump;
    iface.processArgsFunc = process_args;
	iface.loadFunc=main_read;
    if (!MACSIO_IFACE_Register(&iface))
        MACSIO_LOG_MSG(Die, ("Failed to register interface \"%s\"", iface_name));

    return 0;
}

static int dummy = register_this_interface();
