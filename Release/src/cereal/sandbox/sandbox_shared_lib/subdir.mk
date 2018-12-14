################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cereal/sandbox/sandbox_shared_lib/base.cpp \
../src/cereal/sandbox/sandbox_shared_lib/derived.cpp 

OBJS += \
./src/cereal/sandbox/sandbox_shared_lib/base.o \
./src/cereal/sandbox/sandbox_shared_lib/derived.o 

CPP_DEPS += \
./src/cereal/sandbox/sandbox_shared_lib/base.d \
./src/cereal/sandbox/sandbox_shared_lib/derived.d 


# Each subdirectory must supply rules for building sources it contributes
src/cereal/sandbox/sandbox_shared_lib/%.o: ../src/cereal/sandbox/sandbox_shared_lib/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -I"C:\Users\Lauri\Desktop\workspace\NeuralM\src\cereal" -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


