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
#ifdef HAVE_MPI
#include <mpi.h>
#endif
namespace{
using ::tensorstore::Context;
using ::tensorstore::DimensionSet;
using ::tensorstore::DimensionIndex;
std::string Bytes(std::vector<unsigned char> values) {
  return std::string(reinterpret_cast<const char*>(values.data()),
                     values.size());
}
}
::nlohmann::json GetJsonSpec(const char* path,const std::vector<int> shape) {
  return {
{"driver", "zarr3"},
      {"kvstore", {{"driver", "file"}, {"path", path}}},
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
static char const *iface_name = "zarr";
static char const *iface_ext = "zarr";

static int use_log = 0;
static int no_collective = 0;
static int no_single_chunk = 0;
static const char *filename;
static tensorstore::TensorStore<> store;
static int show_errors = 0;


static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}
/*
static void *CreateMyFile(
		const char *fname,
		const char *nsname,
		void *userdata){
	::nlohmann::json json_spec = GetJsonSpec();
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode:open, tensorstore::ReadWriteMode::read_write);
return (void *) store_result;
}

static void *OpenMyFile( 
		const char *fname,
		const char *nsname,
		MACSIO_MIF_ioFlags_t ioFlags,
		 void *userdata){
	::nlohmann::json json_spec = GetJsonSpec();
auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode:open, tensorstore::ReadWriteMode:read_write);	
return (void *) store_result;
*/
static void main_dump(int argi, int argc, char **argv, json_object *main_obj, int dumpn, double dumpt) {
MACSIO_TIMING_GroupMask_t main_dump_sif_grp = MACSIO_TIMING_GroupMask("main_dump_sif");
MACSIO_TIMING_TimerId_t main_dump_sif_tid;
double timer_dt;
    char fileName[256];
 sprintf(fileName, "zarr_%03d",dumpn);
 json_object *global_log_dims_array = json_object_path_get_array(main_obj, "problem/global/LogDims");
const char *json_str7 = json_object_to_json_string(global_log_dims_array );
 std::cout << "dims: " << json_str7 << std::endl;
 int array_len = json_object_array_length(global_log_dims_array);
std::vector<int> shape;
for (int i = 0; i < array_len; ++i) {
            struct json_object *element_obj = json_object_array_get_idx(global_log_dims_array, i);
            int element_value = json_object_get_int(element_obj);
            shape.push_back(element_value);
        }


 ::nlohmann::json json_spec = GetJsonSpec(fileName,shape);
 auto context = Context::Default();
        
 
	
  // Create the store.
 main_dump_sif_tid = MT_StartTimer("createtime", main_dump_sif_grp, dumpn);
 auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create |tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read_write).result();
 timer_dt = MT_StopTimer(main_dump_sif_tid); 
 if (!store_result.ok()) {
    std::cerr << "Failed to create store: " << store_result.status() << std::endl;
  }
  auto store = *store_result;
 
//try to write only 1 dataset
json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
json_object *part_obj = json_object_array_get_idx(part_array, 0);
json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
const char *json_str1 = json_object_to_json_string(extarr_obj );
std::cerr<<"data0: "<<json_str1[0]<<std::endl;
int array_size = json_object_array_length(var_obj);
    std::cout << "JSON array size: " << array_size << std::endl;

	std::vector<int16_t> parsed_data;
	    std::istringstream iss(json_str1);
	        char ch;
		    int value;
		    while (iss >> ch) {
			            if (std::isdigit(ch) || (ch == '-' && std::isdigit(iss.peek()))) {
					                iss.putback(ch);
							            iss >> value;
								                parsed_data.push_back(static_cast<int16_t>(value));
										        }
				        }
		    std::cout << "parsed_size: " << parsed_data.size() << std::endl;
		    std::vector<int16_t> data_to_use(parsed_data.begin() + 4, parsed_data.end());
		     const tensorstore::Index rows = shape[0];
		     const tensorstore::Index cols = shape[1];
auto array = tensorstore::AllocateArray<int16_t>({rows,cols});
for (tensorstore::Index i = 0; i < rows; ++i) {
        for (tensorstore::Index j = 0; j < cols; ++j) {
            array(i, j) = data_to_use[i * cols + j];
        }
    }
 main_dump_sif_tid = MT_StartTimer("ZARRwrite", main_dump_sif_grp, dumpn);
auto write_result = tensorstore::Write(array, store).result();
timer_dt = MT_StopTimer(main_dump_sif_tid);
if (!write_result.ok()) {std::cerr << "Error writing to TensorStore:";}

main_dump_sif_tid = MT_StartTimer("ZARRREAD", main_dump_sif_grp, dumpn);
auto read_result = tensorstore::Read(store).result();
timer_dt = MT_StopTimer(main_dump_sif_tid);

auto dynamic_array = read_result.value();
auto rank_cast_array = tensorstore::StaticRankCast<2>(dynamic_array).value();
    auto typed_array = tensorstore::StaticDataTypeCast<int16_t>(rank_cast_array).value();
/*
std::cout << "Data read from TensorStore:\n";
	        for (tensorstore::Index i = 0; i < rows; ++i) {
			        for (tensorstore::Index j = 0; j < cols; ++j) {
					            std::cout << typed_array(i, j)<< " ";
						            }
				        std::cout << std::endl;
					    }
		/*
auto read_array = tensorstore::Read(store).value();

	    std::cout << "Data read from TensorStore:\n";
	        for (tensorstore::Index i = 0; i < rows; ++i) {
			        for (tensorstore::Index j = 0; j < cols; ++j) {
					read_array(tensorstore::Index(i), tensorstore::Index(j)) << " ";
             }					
			       	}
				        std::cout << std::endl;
					    }

/*
for (tensorstore::Index i = 0; i < parsed_data.size(); ++i) { 
	std::cout <<array(i);}
printf("\n");


		    /*
		    auto array = tensorstore::MakeArray<int16_t>(parsed_data);
		auto spec = tensorstore::Spec::FromJson({
				        {"driver", "zarr"},
					        {"kvstore", {{"driver", "memory"}}},
						        {"metadata", {
							            {"dtype", "<i2"},
								                {"shape", {parsed_data.size()}}
										        }}
											    }).value();


		auto store_result = tensorstore::Open(spec, context, tensorstore::OpenMode::create).result();
		tensorstore::TensorStore<int16_t> store = store_result.value();
		auto write_result = tensorstore::Write(array, store).result();
		if (!write_result.ok()) {std::cerr << "Error writing to TensorStore: " ;}


    /*
    int16_t* parsed_data = NULL;
    size_t count = 0;
    size_t capacity = 1000; 
    parsed_data = (int16_t*)malloc(capacity * sizeof(int16_t));
    const char* ptr = json_str1;
   while (*ptr != '\0') {
	   if (isdigit(*ptr) || (*ptr == '-' && isdigit(*(ptr + 1)))) {
		  if (count >= capacity) {
			  capacity += 100;
			 int16_t* new_data = (int16_t*)realloc(parsed_data, capacity * sizeof(int16_t));
			if (!new_data) {
		       		free(parsed_data);	       
    				}
			parsed_data = new_data;}
		  parsed_data[count++] = (int16_t)strtol(ptr, (char**)&ptr, 10);
		   } else {
			               ++ptr;
				               }
	       }
for (size_t i = 0; i < count; ++i) {
	        printf("%d ", parsed_data[i]);
		    }
printf("\n");
   tensorstore::Array<int16_t,1> array(parsed_data.data(),{parsed_data.size()});
   auto write_result = tensorstore::Write(array, store).result();
   if (!write_result.ok()) {
	               std::cerr << "Error writing to TensorStore: " ;}

    
    
    
    
    
    
    
    
    
    
    /*
	int16_t parsed_data[];
        std::istringstream iss(json_str1);
        char ch;
	int value;
	while (iss >> ch) {
		if (std::isdigit(ch) || (ch == '-' && std::isdigit(iss.peek()))) {
		            iss.putback(ch);
  	                iss >> value;
			parsed_data.push_back(static_cast<int16_t>(value));
			            }
			    }

auto array = tensorstore::MakeArray(parsed_data);
auto write_result = tensorstore::Write(array, store).result();
if (!write_result.ok()) {
	        std::cerr << "Error writing to TensorStore: ";}
/*
std::vector<int> parsed_data;
    int value;
        const char* ptr = json_str1;
	    while (*ptr != '\0') {
		            if (std::isdigit(*ptr) || (*ptr == '-' && std::isdigit(*(ptr + 1)))) {
				                if (sscanf(ptr, "%d", &value) == 1) {
							                parsed_data.push_back(value);
									            }
						            while (std::isdigit(*ptr) || *ptr == '-') {
								                    ++ptr;
										                }
							            } else {
									                ++ptr;
											        }
			        }
auto write_result = tensorstore::Write( tensorstore::MakeArray<int16_t>(parsed_data), store).result();

if (!write_result.ok()) {
	  std::cerr << "Error writing data to TensorStore: " << write_result.status() << std::endl;
} else {
	  std::cout << "Data written to TensorStore successfully!" << std::endl;
}
/*
const char *json_str1 = json_object_to_json_string(extarr_obj );
    std::cerr<<"data: "<<json_str1<<std::endl;
json_object *part_obj = json_object_array_get_idx(part_array, 0);
void const *buf = 0;
json_object *orstore::MakeArray<int16_t>({{1, 2, 3}, {4, 5, 6}}),ars_array = json_object_path_get_array(part_obj, "Vars");
json_object *var_obj = json_object_array_get_idx(vars_array, 0);
json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
buf = json_object_extarr_data(extarr_obj);

const char *json_str1 = json_object_to_json_string(buf );
    std::cerr<<"buf: "<<json_str1<<std::endl;


  
  // Write to the store.
  auto write_result = tensorstore::Write(
      tensorstore::MakeArray<int16_t>({{1, 2, 3}, {4, 5, 6}}),
      store | tensorstore::AllDims().TranslateSizedInterval({9, 8}, {2, 3})).result();

  if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << write_result.status() << std::endl;
  }

  // Read from the store.
  auto read_result = tensorstore::Read<tensorstore::zero_origin>(
      store | tensorstore::AllDims().TranslateSizedInterval({9, 7}, {3, 5})).result();

  if (!read_result.ok()) {
    std::cerr << "Failed to read from store: " << read_result.status() << std::endl;
  }

  // Print the read result.
  auto array = *read_result;
  for (const auto& row : array) {
    for (const auto& elem : row) {
      std::cout << elem << " ";
    }
    std::cout << std::endl;
  }*/
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
