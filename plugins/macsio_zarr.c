#include<stdio.h>

#include <macsio_clargs.h>
#include <macsio_iface.h>
#include <macsio_log.h>
#include <macsio_main.h>
#include <macsio_mif.h>
#include <macsio_timing.h>
#include <macsio_utils.h>

#include <json-cwx/json.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

static char const *iface_name = "zarr";
static char const *iface_ext = "json";
static int json_as_html = 0;
static int my_opt_one;

static int process_args(
        int argi,
        int argc,
        char *argv[]
    )
{
const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};
    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
            "--is_alive","",
            "just get a int and return it",
            &my_opt_one,
            MACSIO_CLARGS_END_OF_ARGS);
        return 0;
}

static void main_dump(
    int argi,               /**< [in] Command-line argument index at which first plugin-specific arg appears */
    int argc,               /**< [in] argc from main */
    char **argv,            /**< [in] argv from main */
    json_object *main_obj,  /**< [in] The main json object representing all data to be dumped */
    int dumpn,              /**< [in] The number/index of this dump. Each dump in a sequence gets a unique,
                                      monotone increasing index starting from 0 */
    double dumpt            /**< [in] The time to be associated with this dump (like a simulation's time) */
){
        printf("This IS alive %d",my_opt_one);
}

static int register_this_interface()
{
    MACSIO_IFACE_Handle_t iface;

    if (strlen(iface_name) >= MACSIO_IFACE_MAX_NAME)
        MACSIO_LOG_MSG(Die, ("Interface name \"%s\" too long", iface_name));

    /* Populate information about this plugin */
    strcpy(iface.name, iface_name);
    strcpy(iface.ext, iface_ext);
    iface.dumpFunc = main_dump;
    iface.processArgsFunc = process_args;
    if (!MACSIO_IFACE_Register(&iface))
        MACSIO_LOG_MSG(Die, ("Failed to register interface \"%s\"", iface_name));

    return 0;
}  

static int const dummy = register_this_interface();


