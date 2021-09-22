CC = g++
CFLAGS_local = -DTEXTOUTPUT -lm -larmadillo -llapack -lopenblas -O3 -fopenmp -ffast-math
CFLAGS_cluster = -DTEXTOUTPUT -lm -I/home/rodrigomh/armadillo/include  -I/opt/lapack/3.9.0/lib64 -I/usr/lib64 -O3 -fopenmp -ffast-math

ITP = src/ITP.cpp
EVOLUTION = src/Evolution.cpp
SRC = src/fields.cpp src/physics.cpp src/math_aux.cpp
SRC_ITP = $(ITP) $(SRC)
SRC_EVOLUTION = $(EVOLUTION) $(SRC)
INCLUDE = ./include/

Evolution_l: $(SRC_EVOLUTION)
	mkdir -p results
	$(CC) $(SRC_EVOLUTION) -I$(INCLUDE) $(CFLAGS_local) -o Evolution
Evolution_c: $(SRC_EVOLUTION)
	mkdir -p results
	$(CC) $(SRC_EVOLUTION) -I$(INCLUDE) $(CFLAGS_cluster) -o Evolution
ITP_l: $(SRC_ITP)
	mkdir -p results
	$(CC) $(SRC_ITP) -I$(INCLUDE) $(CFLAGS_local) -o ITP
ITP_c: $(SRC_ITP)
	mkdir -p results
	$(CC) $(SRC_ITP) -I$(INCLUDE) $(CFLAGS_cluster) -o ITP

