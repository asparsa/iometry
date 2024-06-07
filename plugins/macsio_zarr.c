#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <json-c/json.h>
#include <tensorstore/tensorstore.h>

#include <macsio_clargs.h>
#include <macsio_iface.h>
#include <macsio_log.h>
#include <macsio_main.h>
#include <macsio_mif.h>
#include <macsio_timing.h>
#include <macsio_utils.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

static char const *iface_name = "tensorstore";
static char const *iface_ext = "tstore";

static int use_log = 0;
static int no_collective = 0;
static int no_single_chunk = 0;
static const char *filename;
static tensorstore::TensorStore<> store;
static int show_errors = 0;

static tensorstore::Result<tensorstore::TensorStore<>> make_store(const char* filename) {
    return tensorstore::Open(
        {
            {"driver", "zarr"},
            {"kvstore", {{"driver", "file"}, {"path", filename}}},
            {"metadata", {{"dtype", "<f8"}, {"shape", {100, 100}}}}
        },
        tensorstore::OpenMode::create | tensorstore::OpenMode::delete_existing
    ).result();
}
static int process_args(int argi, int argc, char *argv[]) {
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show low-level TensorStore errors",
            &show_errors,
        "--no_collective", "",
            "Use independent, not collective, I/O calls in SIF mode.",
            &no_collective,
        "--no_single_chunk", "",
            "Do not single chunk the datasets (currently ignored).",
            &no_single_chunk,
           MACSIO_CLARGS_END_OF_ARGS);

    return 0;
}
static void main_dump_sif(json_object *main_obj, int dumpn, double dumpt) {
    MACSIO_TIMING_GroupMask_t main_dump_sif_grp = MACSIO_TIMING_GroupMask("main_dump_sif");
    MACSIO_TIMING_TimerId_t main_dump_sif_tid;
    double timer_dt;

#ifdef HAVE_MPI
    int ndims;
    int i, v, p;
    char const *mesh_type = json_object_path_get_string(main_obj, "clargs/part_type");
    char fileName[256];
    int use_part_count;

    sprintf(fileName, "%s_tstore_%03d.%s",
        json_object_path_get_string(main_obj, "clargs/filebase"),
        dumpn,
        json_object_path_get_string(main_obj, "clargs/fileext"));

    MACSIO_UTILS_RecordOutputFiles(dumpn, fileName);
    main_dump_sif_tid = MT_StartTimer("TensorStore create", main_dump_sif_grp, dumpn);
    auto store_result = make_store(fileName);
    if (!store_result.ok()) {
        fprintf(stderr, "Error creating TensorStore: %s\n", store_result.status().message().c_str());
        return;
    }
    store = store_result.value();
    timer_dt = MT_StopTimer(main_dump_sif_tid);

    ndims = json_object_path_get_int(main_obj, "clargs/part_dim");
    json_object *global_log_dims_array = json_object_path_get_array(main_obj, "problem/global/LogDims");

    std::vector<tensorstore::Index> shape(ndims);
    for (i = 0; i < ndims; i++) {
        shape[ndims - 1 - i] = JsonGetInt(global_log_dims_array, "", i);
    }

    std::vector<double> data(shape[0] * shape[1], 0); // Adjust based on actual dimensions and data type
    json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
    json_object *first_part_obj = json_object_array_get_idx(part_array, 0);
    json_object *first_part_vars_array = json_object_path_get_array(first_part_obj, "Vars");

    for (v = 0; v < json_object_array_length(first_part_vars_array); v++) {
        json_object *var_obj = json_object_array_get_idx(first_part_vars_array, v);
        char const *varName = json_object_path_get_string(var_obj, "name");

        auto array_store = store | tensorstore::IndexSlice(0, tensorstore::Dims(0, 1));
        tensorstore::Write(data, array_store).result();

        use_part_count = (int) ceil(json_object_path_get_double(main_obj, "clargs/avg_num_parts"));
        for (p = 0; p < use_part_count; p++) {
            json_object *part_obj = json_object_array_get_idx(part_array, p);
            json_object *var_obj = 0;

            if (part_obj) {
                json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
                json_object *mesh_obj = json_object_path_get_object(part_obj, "Mesh");
                json_object *var_obj = json_object_array_get_idx(vars_array, v);
                json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
                const double *buf = static_cast<const double*>(json_object_extarr_data(extarr_obj));

                tensorstore::Write(buf, array_store).result();
            }
        }
    }
#endif
}

/*! \brief Main dump callback for TensorStore plugin */
static void main_dump(int argi, int argc, char **argv, json_object *main_obj, int dumpn, double dumpt) {
  tensorstore::Context context = tensorstore::Context::Default();

  std::string path = "C:/Dev/ITKIOOMEZarrNGFF/v0.4/cyx.ome.zarr/s0";

  auto openFuture =
    tensorstore::Open({ { "driver", "zarr" }, { "kvstore", { { "driver", "file" }, { "path", path } } } },
                      context,
                      tensorstore::OpenMode::open,
                      tensorstore::RecheckCached{ false },
                      tensorstore::ReadWriteMode::read);

  auto result = openFuture.result();
  if (result.ok())
  {
    std::cout << "status OK";
    auto store = result.value();
    std::cout << store.domain().shape();
  }
  else
  {
    std::cout << "status BAD\n" << result.status();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
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
