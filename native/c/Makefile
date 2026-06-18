# Build the membench native kernel shared library.
#
# Output: build/libmembench_native.{dylib,so}. The Python layer (membench.
# retrieval._native) discovers it here and falls back to NumPy if it is absent,
# so building is optional — it only matters for large sweeps.

CC ?= cc
CFLAGS ?= -O3 -ffast-math -fPIC -std=c11 -Wall -Wextra
UNAME := $(shell uname)

ifeq ($(UNAME),Darwin)
  EXT := dylib
  SHARED := -dynamiclib
else
  EXT := so
  SHARED := -shared
endif

LIB := build/libmembench_native.$(EXT)

.PHONY: all clean
all: $(LIB)

$(LIB): membench_native.c membench_native.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SHARED) -o $@ membench_native.c

clean:
	rm -rf build
