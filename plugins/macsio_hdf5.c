/*
Copyright (c) 2015, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
Written by Mark C. Miller

LLNL-CODE-676051. All rights reserved.

This file is part of MACSio

Please also read the LICENSE file at the top of the source code directory or
folder hierarchy.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License (as published by the Free Software
Foundation) version 2, dated June 1991.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA 02111-1307 USA
*/
#include <random>
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

#ifdef HAVE_MPI
#include <mpi.h>
#endif

// #ifdef HAVE_SILO
// #include <silo.h> /* for the Silo block based VFD option */
// #endif

#include <H5pubconf.h>
#include <hdf5.h>

/*! \brief H5Z-ZFP generic interface for setting rate mode */
#define H5Pset_zfp_rate_cdata(R, N, CD)          \
do { if (N>=4) {double *p = (double *) &CD[2];   \
CD[0]=CD[1]=CD[2]=CD[3]=0;                       \
CD[0]=1; *p=R; N=4;}} while(0)

/*! \brief H5Z-ZFP generic interface for setting precision mode */
#define H5Pset_zfp_precision_cdata(P, N, CD)  \
do { if (N>=3) {CD[0]=CD[1]=CD[2];            \
CD[0]=2;                 \
CD[2]=P; N=3;}} while(0)

/*! \brief H5Z-ZFP generic interface for setting accuracy mode */
#define H5Pset_zfp_accuracy_cdata(A, N, CD)      \
do { if (N>=4) {double *p = (double *) &CD[2];   \
CD[0]=CD[1]=CD[2]=CD[3]=0;                       \
CD[0]=3; *p=A; N=4;}} while(0)

/*! \brief H5Z-ZFP generic interface for setting expert mode */
#define H5Pset_zfp_expert_cdata(MiB, MaB, MaP, MiE, N, CD) \
do { if (N>=6) { CD[0]=CD[1]=CD[2]=CD[3]=CD[4]=CD[5]=0;    \
CD[0]=4;                                 \
CD[2]=MiB; CD[3]=MaB; CD[4]=MaP;                           \
CD[5]=(unsigned int)MiE; N=6;}} while(0)

/*!
\addtogroup plugins
@{
*/

/*!
\defgroup MACSIO_PLUGIN_HDF5 MACSIO_PLUGIN_HDF5
@{
*/
#define NUM_POINTS 2048
#define NUM_HUGE 4096
/*! \brief name of this plugin */
static char const *iface_name = "hdf5";

/*! \brief file extension for files managed by this plugin */
static char const *iface_ext = "h5";

static int use_log = 0; /**< Use HDF5's logging fapl */
static int no_collective = 0; /**< Use HDF5 independent (e.g. not collective) I/O */
static int no_single_chunk = 0; /**< disable single chunking */
static int chunk1 = 1;
static int chunk2 = 1;
static int chunk3 = 1; 
static int silo_block_size = 0; /**< block size for silo block-based VFD */
static int silo_block_count = 0; /**< block count for silo block-based VFD */
static int sbuf_size = -1; /**< HDF5 library sieve buf size */
static int mbuf_size = -1; /**< HDF5 library meta blocck size */
static int rbuf_size = -1; /**< HDF5 library small data block size */
static int lbuf_size = 0;  /**< HDF5 library log flags */
static const char *filename;
static hid_t fid;
static hid_t dspc = -1;
static int show_errors = 0;
static char compression_alg_str[64];
static char compression_params_str[512];
static char read_type[64];
static char read_sample[64];
static char newpath[64];
/*! \brief create HDF5 library file access property list */
static hid_t make_fapl()
{
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    herr_t h5status = 0;

    if (sbuf_size >= 0)
        h5status |= H5Pset_sieve_buf_size(fapl_id, sbuf_size);

    if (mbuf_size >= 0)
        h5status |= H5Pset_meta_block_size(fapl_id, mbuf_size);

    if (rbuf_size >= 0)
        h5status |= H5Pset_small_data_block_size(fapl_id, mbuf_size);

#if 0
    if (silo_block_size && silo_block_count)
    {
        h5status |= H5Pset_fapl_silo(fapl_id);
        h5status |= H5Pset_silo_block_size_and_count(fapl_id, (hsize_t) silo_block_size,
            silo_block_count);
    }
    else
    if (use_log)
    {
        int flags = H5FD_LOG_LOC_IO|H5FD_LOG_NUM_IO|H5FD_LOG_TIME_IO|H5FD_LOG_ALLOC;

        if (lbuf_size > 0)
            flags = H5FD_LOG_ALL;

        h5status |= H5Pset_fapl_log(fapl_id, "macsio_hdf5_log.out", flags, lbuf_size);
    }
#endif

    {
        H5AC_cache_config_t config;

        /* Acquire a default mdc config struct */
        config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
        H5Pget_mdc_config(fapl_id, &config);
#define MAINZER_PARAMS 1
#if MAINZER_PARAMS
        config.set_initial_size = (hbool_t) 1;
        config.initial_size = 16 * 1024;
        config.min_size = 8 * 1024;
        config.epoch_length = 3000;
        config.lower_hr_threshold = 0.95;
#endif
        H5Pset_mdc_config(fapl_id, &config);
    }

    if (h5status < 0)
    {
        if (fapl_id >= 0)
            H5Pclose(fapl_id);
        return 0;
    }

    return fapl_id;
}

/*!
\brief Utility to parse compression string command-line args

Does a case-insensitive search of \c src_str for \c token_to_match
not including the trailing format specifier. Upon finding a match, 
performs a scanf at the location of the match into temporary 
memory confirming the scan will actually succeed. Then, performs
the scanf again, storying the result to the memory indicated in
\c val_ptr.

\returns 0 on error, 1 on success
*/
static int
get_tokval(
    char const *src_str, /**< CL arg string to be parsed */
    char const *token_to_match, /**< a token in the string to be matched including a trailing scanf format specifier */
    void *val_ptr /**< Pointer to memory where parsed value should be placed */
)
{
    int toklen;
    char dummy[16];
    void *val_ptr_tmp = &dummy[0];

    if (!src_str) return 0;
    if (!token_to_match) return 0;

    toklen = strlen(token_to_match)-2;

    if (strncasecmp(src_str, token_to_match, toklen))
        return 0;

    /* First, check matching sscanf *without* risk of writing to val_ptr */
    if (sscanf(&src_str[toklen], &token_to_match[toklen], val_ptr_tmp) != 1)
        return 0;

    sscanf(&src_str[toklen], &token_to_match[toklen], val_ptr);
    return 1;
}

/*!
\brief create HDF5 library dataset creation property list

If the dataset size is below the \c minsize threshold, no special
storage layout or compression action is taken.

Chunking is initially set to \em single-chunk. However, for szip
compressor, chunking can be set by command-line arguments.

*/
static hid_t
make_dcpl(
    char const *alg_str, /**< compression algorithm string */
    char const *params_str, /**< compression params string */
    hid_t space_id, /**< HDF5 dataspace id for the dataset */
    hid_t dtype_id /**< HDF5 datatype id for the dataset */
)
{
    int shuffle = -1;
    int minsize = -1;
    int gzip_level = -1;
    int zfp_precision = -1;
    unsigned int szip_pixels_per_block = 0;
    float zfp_rate = -1;
    float zfp_accuracy = -1;
    char szip_method[64], szip_chunk_str[64];
    char *token, *string, *tofree;
    int ndims;
    hsize_t dims[4], maxdims[4];
    hid_t retval = H5Pcreate(H5P_DATASET_CREATE);
	H5Sget_simple_extent_dims(space_id, dims, maxdims);
	ndims = H5Sget_simple_extent_ndims(space_id);
    szip_method[0] = '\0';
    szip_chunk_str[0] = '\0';
    //nu_chunk=4;
    /* Initially, set contiguous layout. May reset to chunked later */
    //if (dims[i] % (nu_chunk/2)!=2) chunk_dims[i]++;}
	if (chunk1 == 1) {
    H5Pset_layout(retval, H5D_CONTIGUOUS);
} else {
    H5Pset_layout(retval, H5D_CHUNKED);

    hsize_t chunk_dims[ndims];
	int tmp1[3]={chunk1,chunk2,chunk3};
	for (int i=0;i<ndims;i++) chunk_dims[i]=tmp1[i];
    H5Pset_chunk(retval, ndims, chunk_dims);
}
    

	if (!alg_str || !strlen(alg_str))
        return retval;


    /* We can make a pass through params string without being specific about
       algorithm because there are presently no symbol collisions there */
    tofree = string = strdup(params_str);
    while ((token = strsep(&string, ",")) != NULL)
    {
        if (get_tokval(token, "minsize=%d", &minsize))
            continue;
        if (get_tokval(token, "shuffle=%d", &shuffle))
            continue;
        if (get_tokval(token, "level=%d", &gzip_level))
            continue;
        if (get_tokval(token, "rate=%f", &zfp_rate))
            continue;
        if (get_tokval(token, "precision=%d", &zfp_precision))
            continue;
        if (get_tokval(token, "accuracy=%f", &zfp_accuracy))
            continue;
        if (get_tokval(token, "method=%s", szip_method))
            continue;
        if (get_tokval(token, "block=%u", &szip_pixels_per_block))
            continue;
        if (get_tokval(token, "chunk=%s", szip_chunk_str))
            continue;
    }
    free(tofree);

    /* check for minsize compression threshold */
    minsize = minsize != -1 ? minsize : 1024;
    if (H5Sget_simple_extent_npoints(space_id) < minsize)
        return retval;

    /*
     * Ok, now handle various properties related to compression
     */
 
    /* Initially, as a default in case nothing else is selected,
       set chunk size equal to dataset size (e.g. single chunk) */

    if (!strncasecmp(alg_str, "gzip", 4))
    {
       // if (shuffle == -1 || shuffle == 1)
         //   H5Pset_shuffle(retval);
        H5Pset_deflate(retval, gzip_level!=-1?gzip_level:9);
    }
 
     if (!strncasecmp(alg_str, "zfp", 3))
    {
        unsigned int cd_values[10];
        int cd_nelmts = 10;

        /* Setup ZFP filter and add to HDF5 pipeline using generic interface. */
        if (zfp_rate != -1)
            H5Pset_zfp_rate_cdata(zfp_rate, cd_nelmts, cd_values);
        else if (zfp_precision != -1)
            H5Pset_zfp_precision_cdata(zfp_precision, cd_nelmts, cd_values);
        else if (zfp_accuracy != -1)
            H5Pset_zfp_accuracy_cdata(zfp_accuracy, cd_nelmts, cd_values);
        else
            H5Pset_zfp_rate_cdata(0.0, cd_nelmts, cd_values); /* to get ZFP library defaults */

        /* Add filter to the pipeline via generic interface */
        if (H5Pset_filter(retval, 32013, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values) < 0)
            MACSIO_LOG_MSG(Warn, ("Unable to set up H5Z-ZFP compressor"));
    }
    else if (!strncasecmp(alg_str, "szip", 4))
    {
#ifdef HAVE_SZIP
        unsigned int method = H5_SZIP_NN_OPTION_MASK;
        int const szip_max_blocks_per_scanline = 128; /* from szip lib */

        if (shuffle == -1 || shuffle == 1)
            H5Pset_shuffle(retval);

        if (szip_pixels_per_block == 0)
            szip_pixels_per_block = 32;
        if (!strcasecmp(szip_method, "ec"))
            method = H5_SZIP_EC_OPTION_MASK;

        H5Pset_szip(retval, method, szip_pixels_per_block);

        if (strlen(szip_chunk_str))
        {
            hsize_t chunk_dims[3] = {0, 0, 0};
            int i, vals[3];
            int nvals = sscanf(szip_chunk_str, "%d:%d:%d", &vals[0], &vals[1], &vals[2]);
            if (nvals == ndims)
            {
                for (i = 0; i < ndims; i++)
                    chunk_dims[i] = vals[i];
            }
            else if (nvals == ndims-1)
            {
                chunk_dims[0] = szip_max_blocks_per_scanline * szip_pixels_per_block;
                for (i = 1; i < ndims; i++)
                    chunk_dims[i] = vals[i-1];
            }
            for (i = 0; i < ndims; i++)
            {
                if (chunk_dims[i] > dims[i]) chunk_dims[i] = dims[0];
                if (chunk_dims[i] == 0) chunk_dims[i] = dims[0];
            }
            H5Pset_chunk(retval, ndims, chunk_dims);
        }
#else
        static int have_issued_warning = 0;
        if (!have_issued_warning)
            MACSIO_LOG_MSG(Warn, ("szip compressor not available in this build"));
        have_issued_warning = 1;
#endif
    }

    return retval;
}

/*!
\brief Process command-line arguments an set local variables */
static int
process_args(
    int argi, /**< argument index to start processing \c argv */
    int argc, /**< \c argc from main */
    char *argv[] /**< \c argv from main */
)
{
    const MACSIO_CLARGS_ArgvFlags_t argFlags = {MACSIO_CLARGS_WARN, MACSIO_CLARGS_TOMEM};

    char *c_alg = compression_alg_str;
    char *c_params = compression_params_str;
    char *r_type=read_type;
	char *r_sample=read_sample;
	char *np=newpath;
    MACSIO_CLARGS_ProcessCmdline(0, argFlags, argi, argc, argv,
        "--show_errors", "",
            "Show low-level HDF5 errors",
            &show_errors,
	    "--chunk_dims %d %d %d", "",
	    "number of chunks for dataset",
	    &chunk1, &chunk2, &chunk3,
	    "--read_type %s", "",
	    "choose between full, random points, hyperslap, overlaping hyperslap",
		&r_type,
	"--read_sample %s","",
	    "Choose between SMALL and Large",
	    &r_sample,
	"--path %s", "",
	"give new path here",
	&np,
        "--compression %s %s", MACSIO_CLARGS_NODEFAULT,
            "The first string argument is the compression algorithm name. The second\n"
            "string argument is a comma-separated set of params of the form\n"
            "'param1=val1,param2=val2,param3=val3. The various algorithm names and\n"
            "their parameter meanings are described below. Note that some parameters are\n"
            "not specific to any algorithm. Those are described first followed by\n"
            "individual algorithm-specific parameters for those algorithms available\n"
            "in the current build.\n"
            "\n"
            "minsize=%d : min. size of dataset (in terms of a count of values)\n"
            "    upon which compression will even be attempted. Default is 1024.\n"
            "shuffle=<int>: Boolean (zero or non-zero) to indicate whether to use\n"
            "    HDF5's byte shuffling filter *prior* to compression. Default depends\n"
            "    on algorithm. By default, shuffling is NOT used for zfp but IS\n"
            "    used with all other algorithms.\n"
            "\n"
            "Available compression algorithms...\n"
            "\n"
            "\"zfp\"\n"
            "    Use Peter Lindstrom's ZFP compression (computation.llnl.gov/casc/zfp)\n"
            "    Note: Whether this compression is available is determined entirely at\n"
            "    run-time using the H5Z-ZFP compresser as a generic filter. This means\n"
            "    all that is necessary is to specify the HDF5_PLUGIN_PATH environnment\n" 
            "    variable with a path to the shared lib for the filter.\n"
            "    The following ZFP options are *mutually*exclusive*. In any command-line\n"
            "    specifying more than one of the following options, only the last\n"
            "    specified will be honored.\n"
            "        rate=%f : target # bits per compressed output datum. Fractional values\n"
            "            are permitted. 0 selects defaults: 4 bits/flt or 8 bits/dbl.\n"
            "            Use this option to hit a target compressed size but where error\n"
            "            varies. OTOH, use one of the following two options for fixed\n"
            "            error but amount of compression, if any, varies.\n"
            "        precision=%d : # bits of precision to preserve in each input datum.\n"
            "        accuracy=%f : absolute error tolerance in each output datum.\n"
            "            In many respects, 'precision' represents a sort of relative error\n"
            "            tolerance while 'accuracy' represents an absolute tolerance.\n"
            "            See http://en.wikipedia.org/wiki/Accuracy_and_precision.\n"
            "\n"
#ifdef HAVE_SZIP
            "\"szip\"\n"
            "    method=%s : specify 'ec' for entropy coding or 'nn' for nearest\n"
            "        neighbor. Default is 'nn'\n"
            "    block=%d : (pixels-per-block) must be an even integer <= 32. See\n"
            "        See H5Pset_szip in HDF5 documentation for more information.\n"
            "        Default is 32.\n"
            "    chunk=%d:%d : colon-separated dimensions specifying chunk size in\n"
            "        each dimension higher than the first (fastest varying) dimension.\n"
            "\n"
#endif
            "\"gzip\"\n"
            "    level=%d : A value in the range [1,9], inclusive, trading off time to\n"
            "        compress with amount of compression. Level=1 results in best speed\n"
            "        but worst compression whereas level=9 results in best compression\n"
            "        but worst speed. Values outside [1,9] are clamped. Default is 9.\n"
            "\n"
            "Examples:\n"
            "    --compression zfp rate=18.5\n"
            "    --compression gzip minsize=1024,level=9\n"
            "    --compression szip shuffle=0,options=nn,pixels_per_block=16\n"
            "\n",
            &c_alg, &c_params,
        "--no_collective", "",
            "Use independent, not collective, I/O calls in SIF mode.",
            &no_collective,
        "--no_single_chunk", "",
            "Do not single chunk the datasets (currently ignored).",
            &no_single_chunk,
        "--sieve_buf_size %d", MACSIO_CLARGS_NODEFAULT,
            "Specify sieve buffer size (see H5Pset_sieve_buf_size)",
            &sbuf_size,
        "--meta_block_size %d", MACSIO_CLARGS_NODEFAULT,
            "Specify size of meta data blocks (see H5Pset_meta_block_size)",
            &mbuf_size,
        "--small_block_size %d", MACSIO_CLARGS_NODEFAULT,
            "Specify threshold size for data blocks considered to be 'small'\n"
            "(see H5Pset_small_data_block_size)",
            &rbuf_size,
        "--log", "",
            "Use logging Virtual File Driver (see H5Pset_fapl_log)",
            &use_log,
#ifdef HAVE_SILO
        "--silo_fapl %d %d", MACSIO_CLARGS_NODEFAULT,
            "Use Silo's block-based VFD and specify block size and block count", 
            &silo_block_size, &silo_block_count,
#endif
           MACSIO_CLARGS_END_OF_ARGS);

    if (!show_errors)
        H5Eset_auto1(0,0);
    return 0;
}

/*! \brief Single shared file implementation of main dump */
static void
main_dump_sif(
    json_object *main_obj, /**< main json data object to dump */
    int dumpn, /**< dump number (like a cycle number) */
    double dumpt /**< dump time */
)
{
    MACSIO_TIMING_GroupMask_t main_dump_sif_grp = MACSIO_TIMING_GroupMask("main_dump_sif");
    MACSIO_TIMING_TimerId_t main_dump_sif_tid;
    double timer_dt;

#ifdef HAVE_MPI
    int ndims;
    int i, v, p;
    char const *mesh_type = json_object_path_get_string(main_obj, "clargs/part_type");
    char fileName[256];
    int use_part_count;

    hid_t h5file_id;
    hid_t fapl_id = make_fapl();
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    hid_t null_space_id = H5Screate(H5S_NULL);
    hid_t fspace_nodal_id, fspace_zonal_id;
    hsize_t global_log_dims_nodal[3];
    hsize_t global_log_dims_zonal[3];

    MPI_Info mpiInfo = MPI_INFO_NULL;

//#warning WE ARE DOING SIF SLIGHTLY WRONG, DUPLICATING SHARED NODES
//#warning INCLUDE ARGS FOR ISTORE AND K_SYM
//#warning INCLUDE ARG PROCESS FOR HINTS
//#warning FAPL PROPS: ALIGNMENT 
#if H5_HAVE_PARALLEL
    H5Pset_fapl_mpio(fapl_id, MACSIO_MAIN_Comm, mpiInfo);
#endif
//#warning FOR MIF, NEED A FILEROOT ARGUMENT OR CHANGE TO FILEFMT ARGUMENT
    /* Construct name for the HDF5 file */
    sprintf(fileName, "%s_hdf5_%03d.%s",
        json_object_path_get_string(main_obj, "clargs/filebase"),
        dumpn,
        json_object_path_get_string(main_obj, "clargs/fileext"));

    MACSIO_UTILS_RecordOutputFiles(dumpn, fileName);
    main_dump_sif_tid = MT_StartTimer("H5Fcreate", main_dump_sif_grp, dumpn);
    h5file_id = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    timer_dt = MT_StopTimer(main_dump_sif_tid);

    /* Create an HDF5 Dataspace for the global whole of mesh and var objects in the file. */
    ndims = json_object_path_get_int(main_obj, "clargs/part_dim");
    json_object *global_log_dims_array =
        json_object_path_get_array(main_obj, "problem/global/LogDims");
    json_object *global_parts_log_dims_array =
        json_object_path_get_array(main_obj, "problem/global/PartsLogDims");
    /* Note that global zonal array is smaller in each dimension by one *ON*EACH*BLOCK*
       in the associated dimension. */
    for (i = 0; i < ndims; i++)
    {
        int parts_log_dims_val = JsonGetInt(global_parts_log_dims_array, "", i);
        global_log_dims_nodal[ndims-1-i] = (hsize_t) JsonGetInt(global_log_dims_array, "", i);
        global_log_dims_zonal[ndims-1-i] = global_log_dims_nodal[ndims-1-i] -
            JsonGetInt(global_parts_log_dims_array, "", i);
    }
    fspace_nodal_id = H5Screate_simple(ndims, global_log_dims_nodal, 0);
    fspace_zonal_id = H5Screate_simple(ndims, global_log_dims_zonal, 0);

    /* Get the list of vars on the first part as a guide to loop over vars */
    json_object *part_array = json_object_path_get_array(main_obj, "problem/parts");
    json_object *first_part_obj = json_object_array_get_idx(part_array, 0);
    json_object *first_part_vars_array = json_object_path_get_array(first_part_obj, "Vars");

    /* Dataset transfer property list used in all H5Dwrite calls */
#if H5_HAVE_PARALLEL
    if (no_collective)
        H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT);
    else
        H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);
#endif


    /* Loop over vars and then over parts */
    /* currently assumes all vars exist on all ranks. but not all parts */
    for (int ss=0; ss<1;ss++) /* -1 start is for Mesh */
    {

//#warning SKIPPING MESH
        /* All ranks skip mesh (coords) for now */

        /* Inspect the first part's var object for name, datatype, etc. */
        json_object *var_obj = json_object_array_get_idx(first_part_vars_array, 0);
        char varName[256];
	sprintf(varName,"constant_%03d",ss);
        char *centering = strdup(json_object_path_get_string(var_obj, "centering"));
        json_object *dataobj = json_object_path_get_extarr(var_obj, "data");
//#warning JUST ASSUMING TWO TYPES NOW. CHANGE TO A FUNCTION
        hid_t dtype_id = H5T_NATIVE_DOUBLE; 
        hid_t fspace_id = H5Scopy(strcmp(centering, "zone") ? fspace_nodal_id : fspace_zonal_id);
        hid_t dcpl_id = make_dcpl(compression_alg_str, compression_params_str, fspace_id, dtype_id);

        /* Create the file dataset (using old-style H5Dcreate API here) */
//#warning USING DEFAULT DCPL: LATER ADD COMPRESSION, ETC.
        
        main_dump_sif_tid = MT_StartTimer("H5Dcreate", main_dump_sif_grp, dumpn);
        hid_t ds_id = H5Dcreate1(h5file_id, varName, dtype_id, fspace_id, dcpl_id); 
        timer_dt = MT_StopTimer(main_dump_sif_tid);
        H5Sclose(fspace_id);
        H5Pclose(dcpl_id);

        /* Loop to make write calls for this var for each part on this rank */
//#warning USE NEW MULTI-DATASET API WHEN AVAILABLE TO AGLOMERATE ALL PARTS INTO ONE CALL
        use_part_count = (int) ceil(json_object_path_get_double(main_obj, "clargs/avg_num_parts"));
        for (p = 0; p < use_part_count; p++)
        {
            json_object *part_obj = json_object_array_get_idx(part_array, p);
            json_object *var_obj = 0;
            hid_t mspace_id = H5Scopy(null_space_id);
            void const *buf = 0;

            fspace_id = H5Scopy(null_space_id);

            /* this rank actually has something to contribute to the H5Dwrite call */
            if (part_obj)
            {
                int i;
                hsize_t starts[3], counts[3];
                json_object *vars_array = json_object_path_get_array(part_obj, "Vars");
                json_object *mesh_obj = json_object_path_get_object(part_obj, "Mesh");
                json_object *var_obj = json_object_array_get_idx(vars_array, 0);
                json_object *extarr_obj = json_object_path_get_extarr(var_obj, "data");
                json_object *global_log_origin_array =
                    json_object_path_get_array(part_obj, "GlobalLogOrigin");
                json_object *global_log_indices_array =
                    json_object_path_get_array(part_obj, "GlobalLogIndices");
                json_object *mesh_dims_array = json_object_path_get_array(mesh_obj, "LogDims");
                for (i = 0; i < ndims; i++)
                {
                    starts[ndims-1-i] =
                        json_object_get_int(json_object_array_get_idx(global_log_origin_array,i));
                    counts[ndims-1-i] =
                        json_object_get_int(json_object_array_get_idx(mesh_dims_array,i));
                    if (!strcmp(centering, "zone"))
                    {
                        counts[ndims-1-i]--;
                        starts[ndims-1-i] -=
                            json_object_get_int(json_object_array_get_idx(global_log_indices_array,i));
                    }
                }

                /* set selection of filespace */
                fspace_id = H5Dget_space(ds_id);
                main_dump_sif_tid = MT_StartTimer("H5Sselect_hyperslab", main_dump_sif_grp, dumpn);
                H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, starts, 0, counts, 0);
                timer_dt = MT_StopTimer(main_dump_sif_tid);

                /* set dataspace of data in memory */
                mspace_id = H5Screate_simple(ndims, counts, 0);
                buf = json_object_extarr_data(extarr_obj);
            }

            main_dump_sif_tid = MT_StartTimer("H5Dwrite", main_dump_sif_grp, dumpn);
            H5Dwrite(ds_id, dtype_id, mspace_id, fspace_id, dxpl_id, buf);
            timer_dt = MT_StopTimer(main_dump_sif_tid);
            H5Sclose(fspace_id);
            H5Sclose(mspace_id);

        }

        H5Dclose(ds_id);
        free(centering);
    }

    H5Sclose(fspace_nodal_id);
    H5Sclose(fspace_zonal_id);
    H5Sclose(null_space_id);
    H5Pclose(dxpl_id);
    H5Pclose(fapl_id);
    H5Fclose(h5file_id);

#endif
}

/*! \brief User data for MIF callbacks */
typedef struct _user_data {
    hid_t groupId; /**< HDF5 hid_t of current group */
} user_data_t;

/*! \brief MIF create file callback for HDF5 MIF mode */
static void *
CreateHDF5File(
    const char *fname, /**< file name */
    const char *nsname, /**< curent task namespace name */
    void *userData /**< user data specific to current task */
)
{
    hid_t *retval = 0;
    hid_t h5File;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fclose_degree(fapl, H5F_CLOSE_SEMI);
    h5File = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (h5File >= 0)
    {
//#warning USE NEWER GROUP CREATION SETTINGS OF HDF5
        if (nsname && userData)
        {
            user_data_t *ud = (user_data_t *) userData;
            ud->groupId = H5Gcreate1(h5File, nsname, 0);
        }
        retval = (hid_t *) malloc(sizeof(hid_t));
        *retval = h5File;
    }
    return (void *) retval;
}

/*! \brief MIF Open file callback for HFD5 plugin MIF mode */
static void *
OpenHDF5File(
    const char *fname, /**< filename */
    const char *nsname, /**< namespace name for current task */
    MACSIO_MIF_ioFlags_t ioFlags, /* io flags */
    void *userData /**< task specific user data for current task */
) 
{
    hid_t *retval;
    hid_t h5File;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fclose_degree(fapl, H5F_CLOSE_SEMI);
    h5File = H5Fopen(fname, ioFlags.do_wr ? H5F_ACC_RDWR : H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (h5File >= 0)
    {
        if (ioFlags.do_wr && nsname && userData)
        {
            user_data_t *ud = (user_data_t *) userData;
            ud->groupId = H5Gcreate1(h5File, nsname, 0);
        }
        retval = (hid_t *) malloc(sizeof(hid_t));
        *retval = h5File;
    }
    return (void *) retval;
}

/*! \brief MIF close file callback for HDF5 plugin MIF mode */
static int
CloseHDF5File( 
    void *file, /**< void* to hid_t of file to cose */
    void *userData /**< task specific user data */
)
{
    const unsigned int obj_flags = H5F_OBJ_LOCAL | H5F_OBJ_DATASET |
        H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR;
    int noo;
    herr_t close_retval;

    if (userData)
    {
        user_data_t *ud = (user_data_t *) userData;
        if (H5Iis_valid(ud->groupId) > 0 && H5Iget_type(ud->groupId) == H5I_GROUP)
            H5Gclose(ud->groupId);
    }

    /* Check for any open objects in this file */
    if (fid == (hid_t)H5F_OBJ_ALL ||
        (H5Iis_valid(fid) > 0) && H5Iget_type(fid) == H5I_FILE)
        noo = H5Fget_obj_count(fid, obj_flags);
    close_retval = H5Fclose(*((hid_t*) file));
    free(file);

    if (noo > 0) return -1;
    return (int) close_retval;
}

/*! \brief Write individual mesh part in MIF mode */
static void
write_mesh_part(
    hid_t h5loc, /**< HDF5 group id into which to write */
    json_object *part_obj /**< JSON object for the mesh part to write */
)
{
//#warning WERE SKPPING THE MESH (COORDS) OBJECT PRESENTLY
    int i;
    json_object *vars_array = json_object_path_get_array(part_obj, "Vars");

    for (i = 0; i < json_object_array_length(vars_array); i++)
    {
        int j;
        hsize_t var_dims[3];
        hid_t fspace_id, ds_id, dcpl_id;
        json_object *var_obj = json_object_array_get_idx(vars_array, i);
        json_object *data_obj = json_object_path_get_extarr(var_obj, "data");
        char const *varname = json_object_path_get_string(var_obj, "name");
        int ndims = json_object_extarr_ndims(data_obj);
        void const *buf = json_object_extarr_data(data_obj);
        hid_t dtype_id = json_object_extarr_type(data_obj)==json_extarr_type_flt64? 
                H5T_NATIVE_DOUBLE:H5T_NATIVE_INT;

        for (j = 0; j < ndims; j++)
            var_dims[j] = json_object_extarr_dim(data_obj, j);

        fspace_id = H5Screate_simple(ndims, var_dims, 0);
        dcpl_id = make_dcpl(compression_alg_str, compression_params_str, fspace_id, dtype_id);
        ds_id = H5Dcreate1(h5loc, varname, dtype_id, fspace_id, dcpl_id); 
        H5Dwrite(ds_id, dtype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(ds_id);
        H5Pclose(dcpl_id);
        H5Sclose(fspace_id);
    }
}

/*! \brief Main dump output for HDF5 plugin MIF mode */
static void
main_dump_mif( 
   json_object *main_obj, /**< main data object to dump */
   int numFiles, /**< MIF file count */
   int dumpn, /**< dump number (like a cycle number) */
   double dumpt /**< dump time */
)
{
    MACSIO_TIMING_GroupMask_t main_dump_mif_grp = MACSIO_TIMING_GroupMask("main_dump_mif");
    MACSIO_TIMING_TimerId_t main_dump_mif_tid;
    double timer_dt;

    int size, rank;
    hid_t *h5File_ptr;
    hid_t h5File;
    hid_t h5Group;
    char fileName[256];
    int i, len;
    int *theData;
    user_data_t userData;
    MACSIO_MIF_ioFlags_t ioFlags = {MACSIO_MIF_WRITE,
        (unsigned)JsonGetInt(main_obj, "clargs/exercise_scr")&0x1};

//#warning MAKE WHOLE FILE USE HDF5 1.8 INTERFACE
//#warning SET FILE AND DATASET PROPERTIES
//#warning DIFFERENT MPI TAGS FOR DIFFERENT PLUGINS AND CONTEXTS
    main_dump_mif_tid = MT_StartTimer("MACSIO_MIF_INIT", main_dump_mif_grp, dumpn);
    MACSIO_MIF_baton_t *bat = MACSIO_MIF_Init(numFiles, ioFlags, MACSIO_MAIN_Comm, 3,
        CreateHDF5File, OpenHDF5File, CloseHDF5File, &userData);
    timer_dt = MT_StopTimer(main_dump_mif_tid);

    rank = json_object_path_get_int(main_obj, "parallel/mpi_rank");
    size = json_object_path_get_int(main_obj, "parallel/mpi_size");

    /* Construct name for the silo file */
    sprintf(fileName, "%s_hdf5_%05d_%03d.%s",
        json_object_path_get_string(main_obj, "clargs/filebase"),
        MACSIO_MIF_RankOfGroup(bat, rank),
        dumpn,
        json_object_path_get_string(main_obj, "clargs/fileext"));

    MACSIO_UTILS_RecordOutputFiles(dumpn, fileName);
    
    h5File_ptr = (hid_t *) MACSIO_MIF_WaitForBaton(bat, fileName, 0);
    h5File = *h5File_ptr;
    h5Group = userData.groupId;

    json_object *parts = json_object_path_get_array(main_obj, "problem/parts");

    for (int i = 0; i < json_object_array_length(parts); i++)
    {
        char domain_dir[256];
        json_object *this_part = json_object_array_get_idx(parts, i);
        hid_t domain_group_id;

        snprintf(domain_dir, sizeof(domain_dir), "domain_%07d",
            json_object_path_get_int(this_part, "Mesh/ChunkID"));
 
        domain_group_id = H5Gcreate1(h5File, domain_dir, 0);

        main_dump_mif_tid = MT_StartTimer("write_mesh_part", main_dump_mif_grp, dumpn);
        write_mesh_part(domain_group_id, this_part);
        timer_dt = MT_StopTimer(main_dump_mif_tid);

        H5Gclose(domain_group_id);
    }

    /* If this is the 'root' processor, also write Silo's multi-XXX objects */
#if 0
    if (rank == 0)
        WriteMultiXXXObjects(main_obj, siloFile, bat);
#endif

    /* Hand off the baton to the next processor. This winds up closing
     * the file so that the next processor that opens it can be assured
     * of getting a consistent and up to date view of the file's contents. */
    main_dump_mif_tid = MT_StartTimer("MACSIO_MIF_HandOffBaton", main_dump_mif_grp, dumpn);
    MACSIO_MIF_HandOffBaton(bat, h5File_ptr);
    timer_dt = MT_StopTimer(main_dump_mif_tid);

    /* We're done using MACSIO_MIF, so finish it off */
    main_dump_mif_tid = MT_StartTimer("MACSIO_MIF_Finish", main_dump_mif_grp, dumpn);
    MACSIO_MIF_Finish(bat);
    timer_dt = MT_StopTimer(main_dump_mif_tid);

}

/*!
\brief Main dump callback for HDF5 plugin

Selects between MIF and SSF output.
*/
static void
main_dump(
    int argi, /**< arg index at which to start processing \c argv */
    int argc, /**< \c argc from main */
    char **argv, /**< \c argv from main */
    json_object *main_obj, /**< main json data object to dump */
    int dumpn, /**< dump number */
    double dumpt /**< dump time */
)
{
    MACSIO_TIMING_GroupMask_t main_dump_grp = MACSIO_TIMING_GroupMask("main_dump");
    MACSIO_TIMING_TimerId_t main_dump_tid;
    double timer_dt;

    int rank, size, numFiles;

//#warning SET ERROR MODE OF HDF5 LIBRARY

    /* Without this barrier, I get strange behavior with Silo's MACSIO_MIF interface */
#ifdef HAVE_MPI
    mpi_errno = MPI_Barrier(MACSIO_MAIN_Comm);
#endif

    /* process cl args */
    process_args(argi, argc, argv);

    rank = json_object_path_get_int(main_obj, "parallel/mpi_rank");
    size = json_object_path_get_int(main_obj, "parallel/mpi_size");

//#warning MOVE TO A FUNCTION
    /* ensure we're in MIF mode and determine the file count */
    json_object *parfmode_obj = json_object_path_get_array(main_obj, "clargs/parallel_file_mode");
    if (parfmode_obj)
    {
        json_object *modestr = json_object_array_get_idx(parfmode_obj, 0);
        json_object *filecnt = json_object_array_get_idx(parfmode_obj, 1);
//#warning ERRORS NEED TO GO TO LOG FILES AND ERROR BEHAVIOR NEEDS TO BE HONORED
        if (!strcmp(json_object_get_string(modestr), "SIF"))
        {
            main_dump_tid = MT_StartTimer("main_dump_sif", main_dump_grp, dumpn);
            main_dump_sif(main_obj, dumpn, dumpt);
            timer_dt = MT_StopTimer(main_dump_tid);
        }
        else
        {
            numFiles = json_object_get_int(filecnt);
            main_dump_tid = MT_StartTimer("main_dump_mif", main_dump_grp, dumpn);
            main_dump_mif(main_obj, numFiles, dumpn, dumpt);
            timer_dt = MT_StopTimer(main_dump_tid);
        }
    }
    else
    {
        char const * modestr = json_object_path_get_string(main_obj, "clargs/parallel_file_mode");
        if (!strcmp(modestr, "SIF"))
        {
            float avg_num_parts = json_object_path_get_double(main_obj, "clargs/avg_num_parts");
            if (avg_num_parts == (float ((int) avg_num_parts)))
            {
                main_dump_tid = MT_StartTimer("main_dump_sif", main_dump_grp, dumpn);
                main_dump_sif(main_obj, dumpn, dumpt);
                timer_dt = MT_StopTimer(main_dump_tid);
            }
            else
            {
//#warning CURRENTLY, SIF CAN WORK ONLY ON WHOLE PART COUNTS
                MACSIO_LOG_MSG(Die, ("HDF5 plugin cannot currently handle SIF mode where "
                    "there are different numbers of parts on each MPI rank. "
                    "Set --avg_num_parts to an integral value." ));
            }
        }
        else if (!strcmp(modestr, "MIFMAX"))
            numFiles = json_object_path_get_int(main_obj, "parallel/mpi_size");
        else if (!strcmp(modestr, "MIFAUTO"))
        {
            /* Call utility to determine optimal file count */
//#warning ADD UTILIT TO DETERMINE OPTIMAL FILE COUNT
        }
        main_dump_tid = MT_StartTimer("main_dump_mif", main_dump_grp, dumpn);
        main_dump_mif(main_obj, numFiles, dumpn, dumpt);
        timer_dt = MT_StopTimer(main_dump_tid);
    }
}
#define NUM_HYPERSLABS 4
#define HYPERSLAB_ROWS 2048
#define HYPERSLAB_COLS 2048
#define S_COLS 10484
static void main_read_multihyper(const char* path,json_object *main_obj, int loadnumber )
{
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_multihyper");
        MACSIO_TIMING_TimerId_t main_read_full_tid;
        double timer_dt;
         printf("multi hyper \n");
         char dataset_name[]="dataset00_0";
	hsize_t offset[NUM_HYPERSLABS][2] = {
        {100, 100},
        {15000, 200},
        {15000, 15000},
        {200, 15000}
    };
    hsize_t count[2] = {HYPERSLAB_ROWS, HYPERSLAB_COLS};
    hsize_t slab_size = HYPERSLAB_ROWS * HYPERSLAB_COLS;

    // Open HDF5 file
    hid_t file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "ERROR: Unable to open file: %s\n", path);
        return;
    }

    // Open dataset
    hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "ERROR: Failed to open dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return;
    }

    // Get file dataspace and dataset dimensions
    hid_t file_space_id = H5Dget_space(dataset_id);
    if (file_space_id < 0) {
        fprintf(stderr, "ERROR: Failed to get dataspace from dataset.\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    int rank = H5Sget_simple_extent_ndims(file_space_id);
    if (rank != 2) {
        fprintf(stderr, "ERROR: Expected 2D dataset, got rank %d.\n", rank);
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    hsize_t dset_dims[2];
    H5Sget_simple_extent_dims(file_space_id, dset_dims, NULL);
    printf("Dataset dims: %llu x %llu\n", dset_dims[0], dset_dims[1]);
	if(!strcmp(read_sample,"SMALL")){
    // Allocate data buffer
    int *data = (int *) malloc(NUM_HYPERSLABS * slab_size * sizeof(int));
    if (!data) {
        fprintf(stderr, "ERROR: malloc failed.\n");
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Create memory dataspace for each read
    hsize_t mem_dims[2] = {HYPERSLAB_ROWS, HYPERSLAB_COLS};
    hid_t mem_space_id = H5Screate_simple(2, mem_dims, NULL);

    // Loop over hyperslabs
    main_read_full_tid = MT_StartTimer("small_read", main_read_full_grp, loadnumber);
    for (int i = 0; i < NUM_HYPERSLABS; i++) {
        //printf("Hyperslab %d: offset = (%llu, %llu)\n", i, offset[i][0], offset[i][1]);

        // Bounds check

        // Copy file dataspace for isolated hyperslab selection
        hid_t single_file_space = H5Scopy(file_space_id);

        // Select hyperslab
        herr_t status = H5Sselect_hyperslab(single_file_space, H5S_SELECT_SET, offset[i], NULL, count, NULL);

        // Compute buffer location for this slab
        int *mem_ptr = data + i * slab_size;
        // Read hyperslab into memory
        status = H5Dread(dataset_id, H5T_NATIVE_INT, mem_space_id, single_file_space, H5P_DEFAULT, mem_ptr);
        if (status < 0) {
            fprintf(stderr, "ERROR: H5Dread failed at hyperslab %d\n", i);
        }

        H5Sclose(single_file_space);
    }
	timer_dt = MT_StopTimer(main_read_full_tid);
    // Cleanup
    free(data);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
}
else{
hsize_t count[2] = {S_COLS, S_COLS};
    hsize_t slab_size = S_COLS*S_COLS;
int *data = (int *) malloc(NUM_HYPERSLABS * slab_size * sizeof(int));
    if (!data) {
        fprintf(stderr, "ERROR: malloc failed.\n");
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Create memory dataspace for each read
    hsize_t mem_dims[2] = {S_COLS, S_COLS};
    hid_t mem_space_id = H5Screate_simple(2, mem_dims, NULL);

    // Loop over hyperslabs
    main_read_full_tid = MT_StartTimer("large_read", main_read_full_grp, loadnumber);
    for (int i = 0; i < NUM_HYPERSLABS; i++) {
	/*
        if (offset[i][0] + count[0] > dset_dims[0] || offset[i][1] + count[1] > dset_dims[1]) {
            printf( "Skipping hyperslab %d: out of bounds\n", i);
            continue;
        }

        herr_t status = H5Sselect_hyperslab(single_file_space, H5S_SELECT_SET, offset[i], NULL, count, NULL);
        if (status < 0) {
            fprintf(stderr, "ERROR: H5Sselect_hyperslab failed at %d\n", i);
            H5Sclose(single_file_space);
            continue;
        }*/
        hid_t single_file_space = H5Scopy(file_space_id);
	int *mem_ptr = data + i * slab_size;
        herr_t status = H5Sselect_hyperslab(single_file_space, H5S_SELECT_SET, offset[i], NULL, count, NULL);
	status = H5Dread(dataset_id, H5T_NATIVE_INT, mem_space_id, single_file_space, H5P_DEFAULT, mem_ptr);
        if (status < 0) {
            fprintf(stderr, "ERROR: H5Dread failed at hyperslab %d\n", i);
        }

        H5Sclose(single_file_space);
    }
        timer_dt = MT_StopTimer(main_read_full_tid);
    // Cleanup
    free(data);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
}
}
static void main_read_overhyper(const char* path,json_object *main_obj, int loadnumber )
{
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_overhyper");
        MACSIO_TIMING_TimerId_t main_read_full_tid;
        double timer_dt;
	char dataset_name[]="dataset00_0";
	hsize_t offset[NUM_HYPERSLABS][2] = {
        {15000, 15000},
        {9000, 9000},
        {9000, 15000},
        {15000, 9000}
    };
    hsize_t count[2] = {HYPERSLAB_ROWS, HYPERSLAB_COLS};
    hsize_t slab_size = HYPERSLAB_ROWS * HYPERSLAB_COLS;

    // Open HDF5 file
    hid_t file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "ERROR: Unable to open file: %s\n", path);
        return;
    }

    // Open dataset
    hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "ERROR: Failed to open dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return;
    }

    // Get file dataspace and dataset dimensions
    hid_t file_space_id = H5Dget_space(dataset_id);
    if (file_space_id < 0) {
        fprintf(stderr, "ERROR: Failed to get dataspace from dataset.\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    int rank = H5Sget_simple_extent_ndims(file_space_id);
    if (rank != 2) {
        fprintf(stderr, "ERROR: Expected 2D dataset, got rank %d.\n", rank);
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    hsize_t dset_dims[2];
    H5Sget_simple_extent_dims(file_space_id, dset_dims, NULL);
    printf("Dataset dims: %llu x %llu\n", dset_dims[0], dset_dims[1]);
	if(!strcmp(read_sample,"SMALL")){
    // Allocate data buffer
    int *data = (int *) malloc(NUM_HYPERSLABS * slab_size * sizeof(int));
    if (!data) {
        fprintf(stderr, "ERROR: malloc failed.\n");
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Create memory dataspace for each read
    hsize_t mem_dims[2] = {HYPERSLAB_ROWS, HYPERSLAB_COLS};
    hid_t mem_space_id = H5Screate_simple(2, mem_dims, NULL);

    // Loop over hyperslabs
    main_read_full_tid = MT_StartTimer("small_read", main_read_full_grp, loadnumber);
    for (int i = 0; i < NUM_HYPERSLABS; i++) {
        hid_t single_file_space = H5Scopy(file_space_id);
        herr_t status = H5Sselect_hyperslab(single_file_space, H5S_SELECT_SET, offset[i], NULL, count, NULL);
        int *mem_ptr = data + i * slab_size;
        status = H5Dread(dataset_id, H5T_NATIVE_INT, mem_space_id, single_file_space, H5P_DEFAULT, mem_ptr);
        if (status < 0) {
            fprintf(stderr, "ERROR: H5Dread failed at hyperslab %d\n", i);
        }

        H5Sclose(single_file_space);
    }
	timer_dt = MT_StopTimer(main_read_full_tid);
    // Cleanup
    free(data);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}
else{
hsize_t count[2] = {S_COLS, S_COLS};
    hsize_t slab_size = S_COLS * S_COLS;
int *data = (int *) malloc(NUM_HYPERSLABS * slab_size * sizeof(int));
    if (!data) {
        fprintf(stderr, "ERROR: malloc failed.\n");
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Create memory dataspace for each read
    hsize_t mem_dims[2] = {S_COLS, S_COLS};
    hid_t mem_space_id = H5Screate_simple(2, mem_dims, NULL);

    // Loop over hyperslabs
    main_read_full_tid = MT_StartTimer("huge_read", main_read_full_grp, loadnumber);
    for (int i = 0; i < NUM_HYPERSLABS; i++) {
        hid_t single_file_space = H5Scopy(file_space_id);
        herr_t status = H5Sselect_hyperslab(single_file_space, H5S_SELECT_SET, offset[i], NULL, count, NULL);
        int *mem_ptr = data + i * slab_size;
        status = H5Dread(dataset_id, H5T_NATIVE_INT, mem_space_id, single_file_space, H5P_DEFAULT, mem_ptr);
        if (status < 0) {
            fprintf(stderr, "ERROR: H5Dread failed at hyperslab %d\n", i);
        }

        H5Sclose(single_file_space);
    }
        timer_dt = MT_StopTimer(main_read_full_tid);
    // Cleanup
    free(data);
    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}
}

static void main_read_hyper(const char* path,json_object *main_obj, int loadnumber )
{
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_hyper");
        MACSIO_TIMING_TimerId_t main_read_full_tid;
        double timer_dt;
         printf("main hyper \n");
         char dataset_name[]="dataset00_0";
        main_read_full_tid = MT_StartTimer("H5Fopen", main_read_full_grp, loadnumber);
        hid_t h5file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        timer_dt = MT_StopTimer(main_read_full_tid);
        hid_t dataset_id, space_id, datatype;
herr_t status;
hsize_t dims[2]; // Adjust based on expected dataset rank.
int rank;

// small hyperslab 
hsize_t start1[2] = {1000, 1000};  
hsize_t count1[2] = {8192, 8192};  

// HUGE hyperslab
hsize_t start2[2] = {1000, 1000};  
hsize_t count2[2] = {20968, 20968};  

main_read_full_tid = MT_StartTimer("H5Dopen", main_read_full_grp, loadnumber);
dataset_id = H5Dopen(h5file_id, dataset_name, H5P_DEFAULT);
timer_dt = MT_StopTimer(main_read_full_tid);

space_id = H5Dget_space(dataset_id);
datatype = H5Dget_type(dataset_id);
rank = H5Sget_simple_extent_ndims(space_id);
H5Sget_simple_extent_dims(space_id, dims, NULL);
printf("Dataset rank: %d, Dimensions: %llu x %llu\n", rank, dims[0], dims[1]);

// Allocate memory for each hyperslab
size_t data_size = H5Tget_size(datatype);
size_t total_size1 = data_size * count1[0] * count1[1];
size_t total_size2 = data_size * count2[0] * count2[1];

void *data1 = malloc(total_size1);
void *data2 = malloc(total_size2);
if (!data1 || !data2) {
    printf("Memory allocation failed.\n");
    free(data1);
    free(data2);
    H5Sclose(space_id);
    H5Tclose(datatype);
    H5Dclose(dataset_id);
    H5Fclose(h5file_id);
    return;
}

// Read first hyperslab
if (!strcmp(read_sample,"SMALL")){
hid_t memspace_id1 = H5Screate_simple(rank, count1, NULL);
H5Sselect_none(space_id);  // Reset previous selection
status = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start1, NULL, count1, NULL);
if (status < 0) {
    printf("Failed to select first hyperslab.\n");
} else {
    main_read_full_tid = MT_StartTimer("small_read", main_read_full_grp, loadnumber);
    status = H5Dread(dataset_id, datatype, memspace_id1, space_id, H5P_DEFAULT, data1);
    timer_dt = MT_StopTimer(main_read_full_tid);
    if (status < 0) {
        printf("Failed to read first hyperslab.\n");
    } else {
        printf("Successfully read first hyperslab.\n");
    }
}
H5Sclose(memspace_id1);
free(data1);
free(data2);
H5Sclose(space_id);
H5Tclose(datatype);
H5Dclose(dataset_id);
H5Fclose(h5file_id);
}else{
// Read second hyperslab
hid_t memspace_id2 = H5Screate_simple(rank, count2, NULL);
H5Sselect_none(space_id);  // Reset previous selection
status = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start2, NULL, count2, NULL);
if (status < 0) {
    printf("Failed to select second hyperslab.\n");
} else {
    main_read_full_tid = MT_StartTimer("huge_read", main_read_full_grp, loadnumber);
    status = H5Dread(dataset_id, datatype, memspace_id2, space_id, H5P_DEFAULT, data2);
    timer_dt = MT_StopTimer(main_read_full_tid);
    if (status < 0) {
        printf("Failed to read second hyperslab.\n");
    } else {
        printf("Successfully read second hyperslab.\n");
    }
}
H5Sclose(memspace_id2);

// Clean up
free(data1);
free(data2);
H5Sclose(space_id);
H5Tclose(datatype);
H5Dclose(dataset_id);
H5Fclose(h5file_id);
}
}


static void main_read_full(const char* path,json_object *main_obj, int loadnumber )
{
        MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_full");
        MACSIO_TIMING_TimerId_t main_read_full_tid;
        double timer_dt;
	 printf("main read full \n");
	 char dataset_name[]="dataset00_0";
        main_read_full_tid = MT_StartTimer("H5Fopen", main_read_full_grp, loadnumber);
        hid_t h5file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        timer_dt = MT_StopTimer(main_read_full_tid);
	hid_t dataset_id, space_id, datatype;	
	herr_t status;
    	hsize_t dims[2]; // Adjust based on expected dataset rank.
    	int rank;
	main_read_full_tid = MT_StartTimer("H5Dopen", main_read_full_grp, loadnumber);
        dataset_id = H5Dopen2(h5file_id, dataset_name, H5P_DEFAULT);
	timer_dt = MT_StopTimer(main_read_full_tid);
	space_id = H5Dget_space(dataset_id);
    	datatype = H5Dget_type(dataset_id);
	rank = H5Sget_simple_extent_ndims(space_id);
	H5Sget_simple_extent_dims(space_id, dims, NULL);
	printf("Dataset rank: %d, Dimensions: %llu x %llu\n", rank, dims[0], dims[1]);
	size_t data_size = H5Tget_size(datatype);
    size_t total_size = data_size;
    for (int i = 0; i < rank; i++) {
        total_size *= dims[i];
    }
    void *data = malloc(total_size);
    if (!data) {
        printf("Memory allocation failed.\n");
        H5Sclose(space_id);
        H5Tclose(datatype);
        H5Dclose(dataset_id);
        H5Fclose(h5file_id);
        return;
    }
main_read_full_tid = MT_StartTimer("reading data", main_read_full_grp, loadnumber);    
status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
timer_dt = MT_StopTimer(main_read_full_tid);
if (status < 0) {
        printf("Failed to read dataset: %s\n", dataset_name);
    } else {
        printf("Successfully read dataset: %s\n", dataset_name);
    }
free(data);
    H5Sclose(space_id);
    H5Tclose(datatype);
    H5Dclose(dataset_id);
    H5Fclose(h5file_id);
}
static void main_read_rand(const char* path,json_object *main_obj, int loadnumber ) 
{


MACSIO_TIMING_GroupMask_t main_read_full_grp = MACSIO_TIMING_GroupMask("main_read_rand");
MACSIO_TIMING_TimerId_t main_read_full_tid;
double timer_dt;
hid_t file_id, dataset_id, dataspace_id;
    herr_t status;
	printf("main read rand \n");
	char dataset_name[]="dataset00_0";
    // Open the file
    main_read_full_tid = MT_StartTimer("H5Fopen", main_read_full_grp, loadnumber);
    file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	timer_dt = MT_StopTimer(main_read_full_tid);
    	if (file_id < 0) {
    printf( "Failed to open file: %s\n", path);
}
	// Open the dataset
    main_read_full_tid = MT_StartTimer("H5Dopen", main_read_full_grp, loadnumber);
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
	timer_dt = MT_StopTimer(main_read_full_tid);
    if (dataset_id < 0) {
    fprintf(stderr, "Failed to open dataset: %s\n", dataset_name);
}
	// Get the dataspace
    dataspace_id = H5Dget_space(dataset_id);
	
    // Get dataset rank and dims
    int rank;
    hsize_t dims[2];
    rank = H5Sget_simple_extent_ndims(dataspace_id);
H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
//printf("Dataset rank: %d, dims: %llu x %llu\n", rank, dims[0], dims[1]);

hsize_t (*coords)[2] = NULL;

if (!strcmp(read_sample, "SMALL")) {
    coords = (hsize_t (*)[2]) malloc(NUM_POINTS * sizeof(hsize_t[2]));
    srand(time(NULL));

    for (int idx = 0; idx < NUM_POINTS; idx++) {
        coords[idx][0] = rand() % 32768;
        coords[idx][1] = rand() % 32768;
    }
hsize_t dims[1] = {NUM_POINTS};
hid_t memspace_id = H5Screate_simple(1, dims, NULL);

    // Select the points
    status = H5Sselect_elements(dataspace_id, H5S_SELECT_SET, NUM_POINTS, (const hsize_t *)coords);

    // Allocate buffer
    int *data = (int *) malloc(NUM_POINTS * sizeof(int));

    // Read data
    main_read_full_tid = MT_StartTimer("Read_data_small", main_read_full_grp, loadnumber);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, memspace_id, dataspace_id, H5P_DEFAULT, data);
    timer_dt = MT_StopTimer(main_read_full_tid);
if (status < 0) {
    printf("Failed to read data\n");
}

    // Clean up
    free(coords);
    free(data);
H5Sclose(memspace_id);


	}else{
coords = (hsize_t (*)[2]) malloc(NUM_HUGE * sizeof(hsize_t[2]));
    srand(time(NULL));
hsize_t dims[1] = {NUM_HUGE};
hid_t memspace_id = H5Screate_simple(1, dims, NULL);


    for (int idx = 0; idx < NUM_HUGE; idx++) {
        coords[idx][0] = rand() %  16384;
        coords[idx][1] = rand() %  16384;
    }

    // Select the points
    status = H5Sselect_elements(dataspace_id, H5S_SELECT_SET, NUM_HUGE, (const hsize_t *)coords);

    // Allocate buffer
    int *data = (int *) malloc(NUM_HUGE * sizeof(int));

    // Read data
    main_read_full_tid = MT_StartTimer("Read_data_small", main_read_full_grp, loadnumber);
    status = H5Dread(dataset_id, H5T_NATIVE_INT, memspace_id, dataspace_id, H5P_DEFAULT, data);
    timer_dt = MT_StopTimer(main_read_full_tid);

    // Clean up
    free(coords);
    free(data);
H5Sclose(memspace_id);

}
H5Dclose(dataset_id);
H5Sclose(dataspace_id);
H5Fclose(file_id);

}
static void main_read(int argi, int argc, char **argv, const char* path, json_object *main_obj,int loadnumber ){
	MACSIO_TIMING_GroupMask_t main_read_grp = MACSIO_TIMING_GroupMask("main_read");
	MACSIO_TIMING_TimerId_t main_read_tid;
    	double timer_dt;

    	int rank, size, numFiles;
	process_args(argi, argc, argv);
	printf("read_type=%S \n",read_type);
	if(read_type == NULL) fprintf(stderr, "read_type is NULL!\n");
	else if(!strcmp(read_type,"FULL")){
	main_read_tid=MT_StartTimer("main_read_FULL",main_read_grp,loadnumber);
	main_read_full(newpath, main_obj,loadnumber);
       	timer_dt= MT_StopTimer(main_read_tid);	
	}
	else if(!strcmp(read_type,"HYPER")){
	 main_read_tid=MT_StartTimer("main_read_hyper",main_read_grp,loadnumber);
        main_read_hyper(newpath, main_obj,loadnumber);
        timer_dt= MT_StopTimer(main_read_tid);
	}
	else if(!strcmp(read_type,"MULTI")){
  	       main_read_tid=MT_StartTimer("main_read_multi",main_read_grp,loadnumber);
       		 main_read_multihyper(newpath, main_obj,loadnumber);
         	timer_dt= MT_StopTimer(main_read_tid);
         }
	else if(!strcmp(read_type,"OVER")){
               main_read_tid=MT_StartTimer("main_read_over",main_read_grp,loadnumber);
                 main_read_overhyper(newpath, main_obj,loadnumber);
                timer_dt= MT_StopTimer(main_read_tid);
         }
	else if(!strncmp(read_type,"RAND",4)) {
               main_read_tid=MT_StartTimer("main_read_rand",main_read_grp,loadnumber);
                 main_read_rand(newpath, main_obj,loadnumber);
                timer_dt= MT_StopTimer(main_read_tid);
         }

}
/*! \brief Function called during static initialization to register the plugin */
static int
register_this_interface()
{
    MACSIO_IFACE_Handle_t iface;

    if (strlen(iface_name) >= MACSIO_IFACE_MAX_NAME)
        MACSIO_LOG_MSG(Die, ("Interface name \"%s\" too long", iface_name));

//#warning DO HDF5 LIB WIDE (DEFAULT) INITITILIAZATIONS HERE

    /* Populate information about this plugin */
    strcpy(iface.name, iface_name);
    strcpy(iface.ext, iface_ext);
    iface.dumpFunc = main_dump;
    iface.loadFunc = main_read;
    iface.processArgsFunc = process_args;

    /* Register custom compression methods with HDF5 library */
    H5dont_atexit();

    /* Register this plugin */
    if (!MACSIO_IFACE_Register(&iface))
        MACSIO_LOG_MSG(Die, ("Failed to register interface \"%s\"", iface_name));

    return 0;
}

/*! \brief Static initializer statement to cause plugin registration at link time

this one statement is the only statement requiring compilation by
a C++ compiler. That is because it involves initialization and non
constant expressions (a function call in this case). This function
call is guaranteed to occur during *initialization* (that is before
even 'main' is called) and so will have the effect of populating the
iface_map array merely by virtue of the fact that this code is linked
with a main.
*/
static int dummy = register_this_interface();

/*!@}*/

/*!@}*/
