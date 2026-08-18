BUILD_DIR := build
INSTATTENTION_DIR := instattention
INSTATTENTION_BUILD_DIR := $(INSTATTENTION_DIR)/build-guest
TARGETS = \
	$(BUILD_DIR)/libcemu.a \
	$(BUILD_DIR)/cemu_benchmark \
	$(BUILD_DIR)/test_fdmfs \
	$(BUILD_DIR)/vadd_example \
	$(BUILD_DIR)/test_indirect \
	$(BUILD_DIR)/test_p2p \
	$(BUILD_DIR)/test_iouring \
	$(BUILD_DIR)/test_sync_breakdown

CC = gcc
CXX = g++
PYTHON ?= python3
PYTHON_EXECUTABLE := $(shell command -v $(PYTHON))
CFLAGS = -Wall -Wextra -Wno-unused-function -pedantic -g
CXXFLAGS = $(CFLAGS)

.PHONY: all kernel instattention instattention-test instattention-client-test instattention-device-test instattention-gpu-test instattention-layout-test instattention-kv-store-test instattention-kv-staging-test instattention-prefill-test instattention-flexgen-prefill-test instattention-flexgen-prefill-gpu-test instattention-flexgen-cemu-gpu-test instattention-attention-test instattention-attention-gpu-test

all: $(BUILD_DIR) $(TARGETS) instattention

kernel:
	$(MAKE) -C kernel all

instattention:
	cmake -S $(INSTATTENTION_DIR) -B $(INSTATTENTION_BUILD_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE="$(PYTHON_EXECUTABLE)"
	cmake --build $(INSTATTENTION_BUILD_DIR) --parallel

instattention-client-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_python_bridge.py

instattention-device-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_cemu_device.py

instattention-layout-test:
	python3 $(INSTATTENTION_DIR)/python/test_kv_layout.py

instattention-kv-store-test:
	CEMU_KV_TEST_DIR="$(NVM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_kv_store.py

instattention-kv-staging-test:
	CEMU_NVM_TEST_DIR="$(NVM_TEST_DIR)" CEMU_FDM_TEST_DIR="$(FDM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_kv_staging.py

instattention-prefill-test:
	CEMU_NVM_TEST_DIR="$(NVM_TEST_DIR)" CEMU_FDM_TEST_DIR="$(FDM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_prefill.py

instattention-flexgen-prefill-test:
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_prefill_runtime.py
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_full_prefill.py

instattention-flexgen-prefill-gpu-test:
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_prefill_runtime.py --cuda
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_full_prefill.py --cuda

instattention-flexgen-cemu-gpu-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_cemu_pipeline.py \
		--nvm-dir "$(if $(NVM_TEST_DIR),$(NVM_TEST_DIR),/mnt/nvme0)" \
		--fdm-dir "$(if $(FDM_TEST_DIR),$(FDM_TEST_DIR),/mnt/fdm0)"

instattention-attention-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_attention.py --nvm-dir "$(NVM_TEST_DIR)" --fdm-dir "$(FDM_TEST_DIR)"

instattention-attention-gpu-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_attention.py --cuda --nvm-dir "$(NVM_TEST_DIR)" --fdm-dir "$(FDM_TEST_DIR)"

instattention-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_kv_layout.py
	CEMU_KV_TEST_DIR="$(NVM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_kv_store.py
	CEMU_NVM_TEST_DIR="$(NVM_TEST_DIR)" CEMU_FDM_TEST_DIR="$(FDM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_kv_staging.py
	CEMU_NVM_TEST_DIR="$(NVM_TEST_DIR)" CEMU_FDM_TEST_DIR="$(FDM_TEST_DIR)" python3 $(INSTATTENTION_DIR)/python/test_prefill.py
	python3 $(INSTATTENTION_DIR)/python/test_python_bridge.py
	python3 $(INSTATTENTION_DIR)/python/test_cemu_device.py

instattention-gpu-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_python_bridge.py --cuda
	python3 $(INSTATTENTION_DIR)/python/test_cemu_device.py --cuda

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	chmod 777 $(BUILD_DIR)

$(BUILD_DIR)/libcemu.a: util.c util.h cemu_ioctl.h
	$(CC) $(CFLAGS) -c -o $(BUILD_DIR)/util.o $<
	ar rcs $@ $(BUILD_DIR)/util.o

$(BUILD_DIR)/test_fdmfs: test_fdmfs.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/vadd_example: vadd_example.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/cemu_benchmark: cemu_benchmark.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/test_iouring: test_iouring.c $(BUILD_DIR)/libcemu.a
	$(CC) $(CFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/test_indirect: test_indirect.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/test_parallel_chunks_threads: test_parallel_chunks_threads.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/test_sync_breakdown: test_sync_breakdown.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -luring

$(BUILD_DIR)/test_p2p: test_p2p.cpp $(BUILD_DIR)/libcemu.a
	$(CXX) $(CXXFLAGS) -o $@ $^ -luring

clean:
	rm -rf $(BUILD_DIR) $(INSTATTENTION_BUILD_DIR)
