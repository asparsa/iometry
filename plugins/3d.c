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
static int dim3=1;
::nlohmann::json readjson(int th, const char* path) {
  return {
      {"driver", "zarr3"},
      {"kvstore", {
	{"driver", "file"},
	 {"path", path},
	{"file_io_concurrency", {{"limit", th}}
	}}},
      {"metadata",
       {{"data_type", "int32"},
       }}};
}
::nlohmann::json writejson(int th, std::vector<int> shape, const char* path, std::vector<int> chunksize) {
  return {
    {"driver", "zarr3"},
    {"kvstore", {
      {"driver", "file"},
      {"path", path},
      {"file_io_concurrency", {{"limit", th}}}
    }},
    {"metadata", {
      {"data_type", "int32"},
      {"shape", shape},
      {"chunk_grid", {
        {"name", "regular"},
        {"configuration", {
          {"chunk_shape", chunksize}
        }}
      }}
    }}
  };
}
::nlohmann::json GetJsonSpec2(int th, std::vector<int> shape,const char* path, std::vector<int> chunksize) {
  return ::nlohmann::json{
        {"driver", "zarr3"},
        {"kvstore",
         {
             {"driver", "file"},
             {"path", path},
         }},
        {"metadata",
         {
             {"data_type", "int32"},
             {"shape", shape},
                {"codecs",
                        {
                                {{"name","gzip"},
                                        {"configuration",{{"level",1}}}}
                }}}
        }};
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
static int shape3=1;
static char read_type[64];
static char read_sample[64];
static char newpath[64];
static int num_dims=2;

static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};
	char *r_type=read_type;
	char *r_sample=read_sample;
	char *r_p=newpath;
    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
	 	"--read_type %s", "",
	    "choose between FULL, RANDOM, hyperslap, overlaping hyperslap",
	    &r_type,
	    "--compression_level %d","",
	    "level of compression with gzip",
	    &level,
	    "--chunk_size %d %d %d","",
	    "size of the chunks",
	    &shape1,&shape2,&shape3,
	    "--num_threads %d","",
	    "number of threads for writing",
	    &num_th,
	    "--read_sample %s","",
	    "Choose between SMALL and Large",
	    &r_sample,
		"--path %s","",
		"write the path here",
		&r_p,
	    "--dims %d %d %d","",
	    "dimension of data",
	    &dim1,&dim2,&dim3,
		"--num_dims %d","",
		"number of dimention",
		&num_dims,
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
/*
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

static auto createdata2d(json_object *main_obj, int type,std::vector<int> shape){
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


static auto createdata1d(json_object *main_obj, int type,std::vector<int> shape){
const Index rows = shape[0];
const Index cols = shape[1];
const Index z=shape[1];
auto array=tensorstore::AllocateArray<int>({rows});

if (type==0){
for (Index i=0; i<rows;i++)
        //for(Index j=0;j<cols;j++)
        //for(Index q=0;q<z;q++)
                array(i)=1;
}
return array;
}

static auto createdata3d(json_object *main_obj, int type,std::vector<int> shape){
const Index rows = shape[0];
const Index cols = shape[1];
const Index z=shape[1];
auto array=tensorstore::AllocateArray<int>({rows,cols,z});

if (type==0){
for (Index i=0; i<rows;i++)
        for(Index j=0;j<cols;j++)
        for(Index q=0;q<z;q++)
                array(i,j,q)=1;
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
std::vector<int> chunksize;
std::vector<int> shape;
//creating the metadata
char fileName[256];
//std::vector<int> chunksize={shape1,shape2};
//auto shape= getshape(main_obj);
//std::vector<int> shape={dim1,dim2};
//printf("dim1=%d dim2=%d",dim1,dim2);
//time to create zarr array

 chunksize={shape1,shape2};
 shape={dim1,dim2};
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata1d(main_obj,0,shape);
timer_dt = MT_StopTimer(main_dump_sif_tid);
if(num_dims==1){
 chunksize={shape1};
 shape={dim1};
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata2d(main_obj,0,shape);
timer_dt = MT_StopTimer(main_dump_sif_tid);
}
else if(num_dims==3) {
chunksize={shape1,shape2,shape3};
shape={dim1,dim2,dim3};
main_dump_sif_tid = MT_StartTimer("data_create_time", main_dump_sif_grp, dumpn);
auto data=createdata3d(main_obj,0,shape);
timer_dt = MT_StopTimer(main_dump_sif_tid);
}
/*
//timing for creating the zarr file
main_dump_sif_tid = MT_StartTimer("Zarr_create_time", main_dump_sif_grp, dumpn);
auto create=CreateFile(fileName,shape,shape).result();
timer_dt = MT_StopTimer(main_dump_sif_tid); 
*/
//for(int i=0;i<1000;i++){
sprintf(fileName, "zarr_%03d",dumpn);
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
	
	::nlohmann::json json_spec = readjson(num_th,path);
	main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open);
        timer_dt = MT_StopTimer(main_read_full_tid);
	int16_t* data = (int16_t*)malloc(16384 * 16384 * sizeof(int16_t));
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
	::nlohmann::json json_spec = readjson(num_th,path);
	main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
	auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
	timer_dt = MT_StopTimer(main_read_full_tid);
        //auto randomread = tensorstore::Read(
          //          store , tensorstore::AllDims().TranslateSizedInterval(
            //                    {9, 7}, {1, 1}));
        if (!strcmp(read_sample,"SMALL")){
		main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
		auto result=tensorstore::Read<tensorstore::zero_origin>(
                    	store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 1000}, {4096, 4096}))
                    	.value();
	timer_dt = MT_StopTimer(main_read_full_tid);
	}
	else{
        main_read_full_tid = MT_StartTimer("read_data_huge", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result2=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                { 10,10,10}, { 760,760,760}))
                    .value();
        timer_dt = MT_StopTimer(main_read_full_tid);
	
    }
	   
	
	
}

static void main_read_multihyper(const char* path,json_object *main_obj, int loadnumber ){
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_random");
         MACSIO_TIMING_TimerId_t main_read_full_tid;
         double timer_dt;
        std::cerr << "reading multi-hyper ";
        ::nlohmann::json json_spec = readjson(num_th,path);
        main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
        timer_dt = MT_StopTimer(main_read_full_tid);
        std::vector<std::array<int, 2>> indices = {{3, 5}, {8, 2}, {1, 4}};
        //auto randomread = tensorstore::Read(
          //          store , tensorstore::AllDims().TranslateSizedInterval(
            //                    {9, 7}, {1, 1}));
        if (!strcmp(read_sample,"SMALL")){
	main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100, 100}, {2048, 2048}))
                    .value();
	auto result3=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {7500, 7500}, {2048, 2048}))
                    .value();
auto result4=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100, 7500}, {2048, 2048}))
                    .value();
auto result5=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {7500, 100}, {2048, 2048}))
                    .value();
        timer_dt = MT_StopTimer(main_read_full_tid);
	}else{
        main_read_full_tid = MT_StartTimer("read_data_huge", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result2=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {10,10,10}, {380,380,380}))
                    .value();
        auto result13=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {10,300,10}, {380,380,380}))
                    .value();
auto result12=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {10,10,300 }, { 380,380,380}))
                    .value();
auto result11=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {300,300,10}, { 380,380,380}))
                    .value();
	timer_dt = MT_StopTimer(main_read_full_tid);
     }    
}
static void main_read_overhyper(const char* path,json_object *main_obj, int loadnumber ){
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_random");
         MACSIO_TIMING_TimerId_t main_read_full_tid;
         double timer_dt;
        std::cerr << "reading overlapping multi-hyper ";
        ::nlohmann::json json_spec = readjson(num_th,path);
        main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
        timer_dt = MT_StopTimer(main_read_full_tid);
        std::vector<std::array<int, 2>> indices = {{3, 5}, {8, 2}, {1, 4}};
        //auto randomread = tensorstore::Read(
          //          store , tensorstore::AllDims().TranslateSizedInterval(
            //                    {9, 7}, {1, 1}));
        if (!strcmp(read_sample,"SMALL")){
        main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 1000}, {2048,2048}))
                    .value();
        auto result3=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {2000, 2000}, {2048,2048}))
                    .value();
auto result4=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {1000, 2000}, {2048,2048}))
                    .value();
auto result5=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {2000, 1000}, {800, 800}))
                    .value();
        timer_dt = MT_StopTimer(main_read_full_tid);
	}else{
        main_read_full_tid = MT_StartTimer("read_data_huge", main_read_full_grp, loadnumber);
        //auto read_result = randomread.result();
        auto result2=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                             	{10,10,10}, {380,380,380}))
                    .value();
        auto result13=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {200,200,200}, {380,380,380}))
                    .value();
auto result12=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {400,400,400}, {380,380,380}))
                    .value();
auto result11=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {600,600,600}, {380,380,380}))
                    .value();
        timer_dt = MT_StopTimer(main_read_full_tid);
     }
}

static void main_read_rand(const char* path,json_object *main_obj, int loadnumber ){
	MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_random");
         MACSIO_TIMING_TimerId_t main_read_full_tid;
         double timer_dt;
        std::cerr << "reading randomly ";
        ::nlohmann::json json_spec = readjson(num_th,path);
        main_read_full_tid = MT_StartTimer("open", main_read_full_grp, loadnumber);
        auto store = tensorstore::Open(json_spec, context, tensorstore::OpenMode::open).result();
        timer_dt = MT_StopTimer(main_read_full_tid);
	srand(time(NULL));
	if (!strcmp(read_sample,"SMALL")){
	int arr[4096];
	int arr2[4096];
	for(int i=0;i<2048;i++){
	arr[i]=rand()%16384;
	arr2[i]=rand()%16384;
}
	auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100, 100}, {1,1})).value();
	main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
	for (int i = 0; i < 4096; i++) {
	result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {arr[i],arr2[i]}, {1,1})).value();
}
timer_dt= MT_StopTimer(main_read_full_tid);
}
else{
	int arr[16384];
        int arr2[16384];
	int arr3[16384];
        for(int i=0;i<16384;i++){
        arr[i]=rand()%1024;
        arr2[i]=rand()%1024;
	arr3[i]=rand()%1024;
}
        auto result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100,100,100}, {1,1,1})).value();
        main_read_full_tid = MT_StartTimer("read_data_small", main_read_full_grp, loadnumber);
        for (int i = 0; i < 32684; i++) {
        result=tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {arr[i],arr2[i],arr[i]}, {1,1,1})).value();
}
timer_dt= MT_StopTimer(main_read_full_tid);
}

}

static void main_read(int argi, int argc, char **argv, const char* path, json_object *main_obj,int loadnumber){
	MACSIO_TIMING_GroupMask_t main_read_grp = MACSIO_TIMING_GroupMask("main_read");
	MACSIO_TIMING_TimerId_t main_read_tid;
    	double timer_dt;
	process_args(argi, argc, argv);
	//printf("DEBUG: read_type = %s\n", read_type);

	if(!strncmp(read_type,"FULL",4)){
	main_read_tid=MT_StartTimer("main_read_FULL",main_read_grp,loadnumber);
	main_read_full(newpath, main_obj,loadnumber);
       	timer_dt= MT_StopTimer(main_read_tid);
	}
	else if(!strncmp(read_type,"HYPER",5)){
		main_read_tid=MT_StartTimer("main_read_hyper",main_read_grp,loadnumber);
		main_read_hyper(newpath, main_obj,loadnumber);
		timer_dt= MT_StopTimer(main_read_tid);
	}
	else if(!strncmp(read_type,"MULTI",5)){
                main_read_tid=MT_StartTimer("main_read_multihyper",main_read_grp,loadnumber);
                main_read_multihyper(newpath, main_obj,loadnumber);
                timer_dt= MT_StopTimer(main_read_tid);
        }
	else if(!strncmp(read_type,"RAND",4)){
	 main_read_tid=MT_StartTimer("main_read_rand",main_read_grp,loadnumber);
                main_read_rand(newpath, main_obj,loadnumber);
                timer_dt= MT_StopTimer(main_read_tid);
}	
	else if(!strncmp(read_type,"OVER",4)){
	main_read_tid=MT_StartTimer("main_read_overlapping",main_read_grp,loadnumber);
                main_read_overhyper(newpath, main_obj,loadnumber);
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
