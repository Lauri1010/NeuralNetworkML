################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/neuralM.cpp 

OBJS += \
./src/neuralM.o 

CPP_DEPS += \
./src/neuralM.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -fopenmp -std=c++14 -I"C:\Users\lauri\git\NeuralNetworkML\Includes\cereal\include" -O3 -Ofast -march=native -mtune=native -ftree-vectorize -ffast-math -frename-registers -floop-nest-optimize -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


