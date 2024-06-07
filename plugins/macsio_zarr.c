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
