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


static char const *iface_name = "zarr";
static char const *iface_ext = "zarr";

static int use_log = 0;
static int no_collective = 0;
static int no_single_chunk = 0;
static const char *filename;
//static tensorstore::TensorStore<> store;
static int show_errors = 0;
::nlohmann::json GetJsonSpec(const char* path,const int* shape,const char* dtype,const int* chunks){
	  return {
		  {"driver", "zarri3"},
  	          {"kvstore", {{"driver", "file"}, {"path", path}}},
                  {"metadata",
	           {
	                 {"data_type", dtype},
   		            {"shape", shape},
			    {"configuration", {
                    {"chunk_shape", shape}
                }}
	  	   }}

	     };
}

static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show TensorStore errors",
            &show_errors,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}
static auto create_zarr( const char* filename,const int* shape, const char* dtype, const int* chunks){
    ::nlohmann::json json_spec = GetJsonSpec(filename,shape,dtype,chunks);
    auto context = Context::Default();
    auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create |tensorstore::OpenMode::open,tensorstore::ReadWriteMode::read_write).result(); 
    return store_result;
}



static void main_dump(
    int argi, /**< arg index at which to start processing \c argv */
    int argc, /**< \c argc from main */
    char **argv, /**< \c argv from main */
    json_object *main_obj, /**< main json data object to dump */
    int dumpn, /**< dump number */
    double dumpt /**< dump time */
){
    MACSIO_TIMING_GroupMask_t main_dump_grp = MACSIO_TIMING_GroupMask("main_dump");
    MACSIO_TIMING_TimerId_t main_dump_tid;
    double timer_dt;

    int rank, size, numFiles;

    process_args(argi, argc, argv);


    rank = json_object_path_get_int(main_obj, "parallel/mpi_rank");
    json_object *parfmode_obj = json_object_path_get_array(main_obj, "clargs/parallel_file_mode");
    const char *json_str1 = json_object_to_json_string(parfmode_obj );
    std::cerr<<"parfmode_obj: "<<json_str1<<std::endl;
    json_object *modestr = json_object_array_get_idx(parfmode_obj, 0);
    json_object *filecnt = json_object_array_get_idx(parfmode_obj, 1);
    //add if condition for MIF later
    main_dump_tid = MT_StartTimer("main_dump_sif", main_dump_grp, dumpn);
    main_dump_sif(main_obj, dumpn, dumpt);
    timer_dt = MT_StopTimer(main_dump_tid);

    
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
