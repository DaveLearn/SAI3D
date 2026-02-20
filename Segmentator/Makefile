CXX ?= g++
FLAGS ?= -std=c++11
LDFLAGS ?=

ifneq ($(CONDA_PREFIX),)
LDFLAGS += -Wl,-rpath,$(CONDA_PREFIX)/lib -L$(CONDA_PREFIX)/lib
endif

main:
	$(CXX) $(FLAGS) $(LDFLAGS) -o segmentator segmentator.cpp tinyply.cpp

clean:
	rm -f *~ *.o segmentator
