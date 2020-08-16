#!/bin/bash

set -eu

./python/main.py
g++ ./c++/main.cpp -g -std=c++11 -fsanitize=address -fsanitize=undefined -Wall -Wextra -Wshadow
./a.out
rm ./a.out
