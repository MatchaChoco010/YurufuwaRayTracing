a.out: main.o
	g++ -Wall -std=c++17 -fopenmp main.o
main.o: main.cpp
	g++ -Wall -std=c++17 -fopenmp -c main.cpp -o main.o
