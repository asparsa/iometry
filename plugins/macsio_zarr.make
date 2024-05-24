

ZARR_BUILD_ORDER=4.0
ZARR_HOME=
ZARR_CFLAGS=
ZARR_LDFLAGS=
ZARR_SOURCES=macsio_zarr.c
PLUGIN_OBJECTS += $(ZARR_SOURCES:.c=.o)
PLUGIN_LDFLAGS += $(ZARR_LDFLAGS)
PLUGIN_LIST += zarr
macsio_zarr.o: ../plugins/macsio_zarr.c
	$(CXX) -c $(zarr_CFLAGS) $(MACSIO_CFLAGS) $(CFLAGS) ../plugins/macsio_zarr.c


