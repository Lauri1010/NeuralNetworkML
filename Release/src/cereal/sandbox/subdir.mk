################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cereal/sandbox/performance.cpp \
../src/cereal/sandbox/sandbox.cpp \
../src/cereal/sandbox/sandbox_json.cpp \
../src/cereal/sandbox/sandbox_rtti.cpp \
../src/cereal/sandbox/sandbox_vs.cpp 

OBJS += \
./src/cereal/sandbox/performance.o \
./src/cereal/sandbox/sandbox.o \
./src/cereal/sandbox/sandbox_json.o \
./src/cereal/sandbox/sandbox_rtti.o \
./src/cereal/sandbox/sandbox_vs.o 

CPP_DEPS += \
./src/cereal/sandbox/performance.d \
./src/cereal/sandbox/sandbox.d \
./src/cereal/sandbox/sandbox_json.d \
./src/cereal/sandbox/sandbox_rtti.d \
./src/cereal/sandbox/sandbox_vs.d 


# Each subdirectory must supply rules for building sources it contributes
src/cereal/sandbox/%.o: ../src/cereal/sandbox/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -I"C:\Users\Lauri\Desktop\workspace\NeuralM\src\cereal" -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


