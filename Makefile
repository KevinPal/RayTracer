CXX	  := nvcc
CXXFLAGS := -g -G -rdc=true
LDFLAGS  := -L/usr/lib -lstdc++ -lm
BUILD	:= ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := mp1

INCLUDE  := -I include/ \
	`pkg-config -cflags glib-2.0` \
	`pkg-config -cflags gtkmm-3.0 | head -c-11`

# the head part removes the pthread dependancy, since nvcc already has
# it and complains if we include it again

LDFLAGS += \
	`pkg-config -libs glib-2.0` \
	`pkg-config -libs gtkmm-3.0`

SRC	  :=					  \
   $(wildcard src/*.cpp)		 \

SRC_CU	  :=					  \
   $(wildcard src/*.cu)		 \

OBJECTS     := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
OBJECTS_CU  := $(SRC_CU:%.cu=$(OBJ_DIR)/%.obj)

DEPENDENCIES \
		 := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(OBJ_DIR)/%.obj: %.cu
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS) $(OBJECTS_CU)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info run clean_run

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all
debug: 
	@cuda-gdb build/apps/mp1

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}	 "
	@echo "[*] Object dir:	  ${OBJ_DIR}	 "
	@echo "[*] Sources:		 ${SRC}		 "
	@echo "[*] Objects:		 ${OBJECTS}	 "
	@echo "[*] Dependencies:	${DEPENDENCIES}"

run: all
run:
	@echo "Running ${TARGET}"
	@./build/apps/$(TARGET)
	@echo "Done running ${TARGET}"

clean_run: clean all run


