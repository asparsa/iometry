#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <json-c/json.h>

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

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "zarr"},
      {"kvstore", {{"driver", "file"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", "<i2"},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
}
}
static char const *iface_name = "tensorstore";
static char const *iface_ext = "tstore";

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
  ::nlohmann::json json_spec = GetJsonSpec();
 auto context = Context::Default();

  // Create the store.
  auto store_result = tensorstore::Open(json_spec, context, tensorstore::OpenMode::create |tensorstore::OpenMode::open,
                                        tensorstore::ReadWriteMode::read_write).result();
  if (!store_result.ok()) {
    std::cerr << "Failed to create store: " << store_result.status() << std::endl;
  }
  auto store = *store_result;

  // Write to the store.
  auto write_result = tensorstore::Write(
      tensorstore::MakeArray<int16_t>({{1, 2, 3}, {4, 5, 6}}),
      store | tensorstore::AllDims().TranslateSizedInterval({9, 8}, {2, 3})).result();

  if (!write_result.ok()) {
    std::cerr << "Failed to write to store: " << write_result.status() << std::endl;
  }
/*
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
