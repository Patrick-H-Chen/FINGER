CXX=g++
CXXFLAGS=-fopenmp -O3 -std=c++14 -fPIC -DNDEBUG -Wall -g -lopenblas 
#EXTRA_INCLUDE_FLAGS=-I./eigen-3.3.9/ -lopenblas -llapack
#EXTRA_INCLUDE_FLAGS=-lopenblas 
ARCHFLAG=-march=native -mavx512vl 

all: go

go: example.cpp
	${CXX} -o go ${CXXFLAGS} example.cpp -I. ${EXTRA_INCLUDE_FLAGS} ${ARCHFLAG}
clean:
	rm -rf *.so *.o go
