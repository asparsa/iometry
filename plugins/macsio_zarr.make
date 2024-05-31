# Copyright (c) 2015, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Mark C. Miller
#
# LLNL-CODE-676051. All rights reserved.
#
# This file is part of MACSio
# 
# Please also read the LICENSE file at the top of the source code directory or
# folder hierarchy.
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2, dated June 1991.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA

# This floating point variable is used to order plugin objects during
# the main link for MACSio to allow dependent libraries that are common
# to multiple plugins to be placed later on the link line. Bigger
# numbers here cause them to appear later in the link line.
TENSORSTORE_BUILD_ORDER = 1.0

ifneq ($(TENSORSTORE_HOME),)

TENSORSTORE_LDFLAGS = -L$(TENSORSTORE_HOME)/lib -ltensorstore -Wl,-rpath,$(TENSORSTORE_HOME)/lib
TENSORSTORE_CFLAGS = -I$(TENSORSTORE_HOME)/include

TENSORSTORE_SOURCES = macsio_tensorstore.cpp

ifneq ($(SZIP_HOME),)
TENSORSTORE_LDFLAGS += -L$(SZIP_HOME)/lib -lsz -Wl,-rpath,$(SZIP_HOME)/lib
TENSORSTORE_CFLAGS += -DHAVE_SZIP
endif

ifneq ($(ZLIB_HOME),)
TENSORSTORE_LDFLAGS += -L$(ZLIB_HOME)/lib
endif

TENSORSTORE_LDFLAGS += -lz -lm

PLUGIN_OBJECTS += $(TENSORSTORE_SOURCES:.cpp=.o)
PLUGIN_LDFLAGS += $(TENSORSTORE_LDFLAGS)
PLUGIN_LIST += tensorstore

endif

macsio_tensorstore.o: ../plugins/macsio_tensorstore.cpp
	$(CXX) -c $(TENSORSTORE_CFLAGS) $(MACSIO_CFLAGS) $(CFLAGS) ../plugins/macsio_tensorstore.cpp

list-tpls-tensorstore:
	@echo "TensorStore library required"

download-tpls-tensorstore:
	@echo "No download required for TensorStore"
