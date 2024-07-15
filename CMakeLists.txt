cmake_minimum_required (VERSION 2.6)
project(mini_rnn LANGUAGES CXX)

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

add_executable(rnn 
    main.cpp 
    RNNDescent.cpp
)
