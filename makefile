a.out: main.o
	g++ -Wall -std=c++17 main.o
main.o: main.cpp
	g++ -Wall -std=c++17 -c main.cpp -o main.o
