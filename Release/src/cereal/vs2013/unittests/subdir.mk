################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cereal/vs2013/unittests/main.cpp 

OBJS += \
./src/cereal/vs2013/unittests/main.o 

CPP_DEPS += \
./src/cereal/vs2013/unittests/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/cereal/vs2013/unittests/%.o: ../src/cereal/vs2013/unittests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -I"C:\Users\Lauri\Desktop\workspace\NeuralM\src\cereal" -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


