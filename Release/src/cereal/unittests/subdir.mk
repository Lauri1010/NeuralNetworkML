################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cereal/unittests/array.cpp \
../src/cereal/unittests/basic_string.cpp \
../src/cereal/unittests/bitset.cpp \
../src/cereal/unittests/boost_variant.cpp \
../src/cereal/unittests/chrono.cpp \
../src/cereal/unittests/complex.cpp \
../src/cereal/unittests/deque.cpp \
../src/cereal/unittests/forward_list.cpp \
../src/cereal/unittests/list.cpp \
../src/cereal/unittests/load_construct.cpp \
../src/cereal/unittests/map.cpp \
../src/cereal/unittests/memory.cpp \
../src/cereal/unittests/memory_cycles.cpp \
../src/cereal/unittests/multimap.cpp \
../src/cereal/unittests/multiset.cpp \
../src/cereal/unittests/pair.cpp \
../src/cereal/unittests/pod.cpp \
../src/cereal/unittests/polymorphic.cpp \
../src/cereal/unittests/portability_test.cpp \
../src/cereal/unittests/portable_binary_archive.cpp \
../src/cereal/unittests/priority_queue.cpp \
../src/cereal/unittests/queue.cpp \
../src/cereal/unittests/set.cpp \
../src/cereal/unittests/stack.cpp \
../src/cereal/unittests/structs.cpp \
../src/cereal/unittests/structs_minimal.cpp \
../src/cereal/unittests/structs_specialized.cpp \
../src/cereal/unittests/tuple.cpp \
../src/cereal/unittests/unordered_loads.cpp \
../src/cereal/unittests/unordered_map.cpp \
../src/cereal/unittests/unordered_multimap.cpp \
../src/cereal/unittests/unordered_multiset.cpp \
../src/cereal/unittests/unordered_set.cpp \
../src/cereal/unittests/user_data_adapters.cpp \
../src/cereal/unittests/valarray.cpp \
../src/cereal/unittests/vector.cpp \
../src/cereal/unittests/versioning.cpp 

OBJS += \
./src/cereal/unittests/array.o \
./src/cereal/unittests/basic_string.o \
./src/cereal/unittests/bitset.o \
./src/cereal/unittests/boost_variant.o \
./src/cereal/unittests/chrono.o \
./src/cereal/unittests/complex.o \
./src/cereal/unittests/deque.o \
./src/cereal/unittests/forward_list.o \
./src/cereal/unittests/list.o \
./src/cereal/unittests/load_construct.o \
./src/cereal/unittests/map.o \
./src/cereal/unittests/memory.o \
./src/cereal/unittests/memory_cycles.o \
./src/cereal/unittests/multimap.o \
./src/cereal/unittests/multiset.o \
./src/cereal/unittests/pair.o \
./src/cereal/unittests/pod.o \
./src/cereal/unittests/polymorphic.o \
./src/cereal/unittests/portability_test.o \
./src/cereal/unittests/portable_binary_archive.o \
./src/cereal/unittests/priority_queue.o \
./src/cereal/unittests/queue.o \
./src/cereal/unittests/set.o \
./src/cereal/unittests/stack.o \
./src/cereal/unittests/structs.o \
./src/cereal/unittests/structs_minimal.o \
./src/cereal/unittests/structs_specialized.o \
./src/cereal/unittests/tuple.o \
./src/cereal/unittests/unordered_loads.o \
./src/cereal/unittests/unordered_map.o \
./src/cereal/unittests/unordered_multimap.o \
./src/cereal/unittests/unordered_multiset.o \
./src/cereal/unittests/unordered_set.o \
./src/cereal/unittests/user_data_adapters.o \
./src/cereal/unittests/valarray.o \
./src/cereal/unittests/vector.o \
./src/cereal/unittests/versioning.o 

CPP_DEPS += \
./src/cereal/unittests/array.d \
./src/cereal/unittests/basic_string.d \
./src/cereal/unittests/bitset.d \
./src/cereal/unittests/boost_variant.d \
./src/cereal/unittests/chrono.d \
./src/cereal/unittests/complex.d \
./src/cereal/unittests/deque.d \
./src/cereal/unittests/forward_list.d \
./src/cereal/unittests/list.d \
./src/cereal/unittests/load_construct.d \
./src/cereal/unittests/map.d \
./src/cereal/unittests/memory.d \
./src/cereal/unittests/memory_cycles.d \
./src/cereal/unittests/multimap.d \
./src/cereal/unittests/multiset.d \
./src/cereal/unittests/pair.d \
./src/cereal/unittests/pod.d \
./src/cereal/unittests/polymorphic.d \
./src/cereal/unittests/portability_test.d \
./src/cereal/unittests/portable_binary_archive.d \
./src/cereal/unittests/priority_queue.d \
./src/cereal/unittests/queue.d \
./src/cereal/unittests/set.d \
./src/cereal/unittests/stack.d \
./src/cereal/unittests/structs.d \
./src/cereal/unittests/structs_minimal.d \
./src/cereal/unittests/structs_specialized.d \
./src/cereal/unittests/tuple.d \
./src/cereal/unittests/unordered_loads.d \
./src/cereal/unittests/unordered_map.d \
./src/cereal/unittests/unordered_multimap.d \
./src/cereal/unittests/unordered_multiset.d \
./src/cereal/unittests/unordered_set.d \
./src/cereal/unittests/user_data_adapters.d \
./src/cereal/unittests/valarray.d \
./src/cereal/unittests/vector.d \
./src/cereal/unittests/versioning.d 


# Each subdirectory must supply rules for building sources it contributes
src/cereal/unittests/%.o: ../src/cereal/unittests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -I"C:\Users\Lauri\Desktop\workspace\NeuralM\src\cereal" -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


