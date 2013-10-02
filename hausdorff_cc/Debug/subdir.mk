################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../adj_path.cpp \
../adj_user_paths.cpp \
../latlon.cpp \
../main.cpp 

OBJS += \
./adj_path.o \
./adj_user_paths.o \
./latlon.o \
./main.o 

CPP_DEPS += \
./adj_path.d \
./adj_user_paths.d \
./latlon.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/paolo/Projects/pathadjacency/trunk/hausdorff_cc/Debug/include -O0 -ffast-math -g3 -Wall -c -fmessage-length=0  -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


