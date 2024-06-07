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
#include "tensorstore/context.h"
#include "tensorstore/open.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

std::string Bytes(std::vector<unsigned char> values) {
  return std::string(reinterpret_cast<const char*>(values.data()),
                     values.size());
}

::nlohmann::json GetJsonSpec() {
  return {
      {"driver", "zarr"},
      {"kvstore", {{"driver", "memory"}, {"path", "prefix/"}}},
      {"metadata",
       {
           {"compressor", {{"id", "blosc"}}},
           {"dtype", "<i2"},
           {"shape", {100, 100}},
           {"chunks", {3, 2}},
       }},
  };
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
  ::nlohmann::json json_spec = GetJsonSpec();

  auto context = Context::Default();
  // Create the store.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                          tensorstore::ReadWriteMode::read_write)
            .result());
    EXPECT_THAT(store.domain().origin(), ::testing::ElementsAre(0, 0));
    EXPECT_THAT(store.domain().shape(), ::testing::ElementsAre(100, 100));
    EXPECT_THAT(store.domain().labels(), ::testing::ElementsAre("", ""));
    EXPECT_THAT(store.domain().implicit_lower_bounds(),
                DimensionSet::FromBools({0, 0}));
    EXPECT_THAT(store.domain().implicit_upper_bounds(),
                DimensionSet::FromBools({1, 1}));

    // Test ResolveBounds.
    auto resolved = ResolveBounds(store).value();
    EXPECT_EQ(store.domain(), resolved.domain());

    // Test ResolveBounds with a transform that swaps upper and lower bounds.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto reversed_dim0,
        store | tensorstore::Dims(0).ClosedInterval(kImplicit, kImplicit, -1));
    auto resolved_reversed_dim0 = ResolveBounds(reversed_dim0).value();
    EXPECT_EQ(reversed_dim0.domain(), resolved_reversed_dim0.domain());

    // Issue a read to be filled with the fill value.
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {9, 7}, {1, 1}))
                    .result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>({{0}})));

    // Issue an out-of-bounds read.
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100, 7}, {1, 1}))
                    .result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));

    // Issue a valid write.
    TENSORSTORE_EXPECT_OK(tensorstore::Write(
        tensorstore::MakeArray<int16_t>({{1, 2, 3}, {4, 5, 6}}),
        store | tensorstore::AllDims().TranslateSizedInterval({9, 8}, {2, 3})));

    // Issue an out-of-bounds write.
    EXPECT_THAT(tensorstore::Write(
                    tensorstore::MakeArray<int16_t>({{1, 2, 3}, {4, 5, 6}}),
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {100, 8}, {2, 3}))
                    .result(),
                MatchesStatus(absl::StatusCode::kOutOfRange));

    // Re-read and validate result.
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {9, 7}, {3, 5}))
                    .result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>(
                    {{0, 1, 2, 3, 0}, {0, 4, 5, 6, 0}, {0, 0, 0, 0, 0}})));
  }

  // Check that key value store has expected contents.
  EXPECT_THAT(
      GetMap(kvstore::Open({{"driver", "memory"}}, context).value()).value(),
      UnorderedElementsAreArray({
          Pair("prefix/.zarray",  //
               ::testing::MatcherCast<absl::Cord>(ParseJsonMatches({
                   {"zarr_format", 2},
                   {"order", "C"},
                   {"filters", nullptr},
                   {"fill_value", nullptr},
                   {"compressor",
                    {{"id", "blosc"},
                     {"blocksize", 0},
                     {"clevel", 5},
                     {"cname", "lz4"},
                     {"shuffle", -1}}},
                   {"dtype", "<i2"},
                   {"shape", {100, 100}},
                   {"chunks", {3, 2}},
                   {"dimension_separator", "."},
               }))),
          Pair("prefix/3.4",    //
               DecodedMatches(  //
                   Bytes({1, 0, 2, 0, 4, 0, 5, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
          Pair("prefix/3.5",    //
               DecodedMatches(  //
                   Bytes({3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0}),
                   tensorstore::blosc::Decode)),
      }));

  // Check that attempting to create the store again fails.
  EXPECT_THAT(
      tensorstore::Open(json_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result(),
      MatchesStatus(absl::StatusCode::kAlreadyExists,
                    "Error opening \"zarr\" driver: "
                    "Error writing \"prefix/\\.zarray\""));

  // Check that create or open succeeds.
  TENSORSTORE_EXPECT_OK(tensorstore::Open(
      json_spec, context,
      tensorstore::OpenMode::create | tensorstore::OpenMode::open,
      tensorstore::ReadWriteMode::read_write));

  // Check that open succeeds.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, tensorstore::OpenMode::open,
                          tensorstore::ReadWriteMode::read_write)
            .result());
    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {9, 7}, {3, 5}))
                    .result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>(
                    {{0, 1, 2, 3, 0}, {0, 4, 5, 6, 0}, {0, 0, 0, 0, 0}})));
  }

  // Check that delete_existing works.
  for (auto transaction_mode :
       {tensorstore::TransactionMode::no_transaction_mode,
        tensorstore::TransactionMode::isolated,
        tensorstore::TransactionMode::atomic_isolated}) {
    tensorstore::Transaction transaction(transaction_mode);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        tensorstore::Open(json_spec, context, transaction,
                          tensorstore::OpenMode::create |
                              tensorstore::OpenMode::delete_existing,
                          tensorstore::ReadWriteMode::read_write)
            .result());

    EXPECT_THAT(tensorstore::Read<tensorstore::zero_origin>(
                    store | tensorstore::AllDims().TranslateSizedInterval(
                                {9, 7}, {3, 5}))
                    .result(),
                ::testing::Optional(tensorstore::MakeArray<int16_t>(
                    {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}})));
    TENSORSTORE_ASSERT_OK(transaction.CommitAsync());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto kvs, kvstore::Open({{"driver", "memory"}}, context).result());
    EXPECT_THAT(
        ListFuture(kvs).value(),
        ::testing::UnorderedElementsAre(MatchesListEntry("prefix/.zarray")));
  }
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

    if (!MACSIO_IFACE_Register(&iface))
        MACSIO_LOG_MSG(Die, ("Failed to register interface \"%s\"", iface_name));

    return 0;
}

static int dummy = register_this_interface();
