CC = g++
CFLAGS_local = -lm -larmadillo -llapack -lopenblas -O3 -fopenmp -ffast-math
CFLAGS_cluster = -lm -I/home/rodrigomh/armadillo/include  -I-I/opt/lapack/3.9.0/lib64 -l/usr/lib64 -O3 -fopenmp -ffast-math

SRC = src/fields.cpp src/physics.cpp src/math_aux.cpp
INCLUDE = ./include/

Evolution_l: src/Evolution.cpp $(SRC)
	$(CC) $(SRC) src/Evolution.cpp $(CFLAGS_local) -o Evolution
Evolution_c: src/Evolution.cpp $(SRC)
	$(CC) $(SRC) src/Evolution.cpp $(CFLAGS_cluster) -o Evolution
ITP_l: src/ITP.cpp $(SRC)
	$(CC) $(SRC) src/ITP.cpp $(CFLAGS_local) -o ITP
ITP_l: src/ITP.cpp $(SRC)
	$(CC) $(SRC) src/ITP.cpp $(CFLAGS_cluster) -o ITP

