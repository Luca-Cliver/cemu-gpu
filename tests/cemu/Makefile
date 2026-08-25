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

.PHONY: all kernel instattention instattention-test instattention-client-test instattention-device-test instattention-gpu-test instattention-layout-test instattention-kv-store-test instattention-kv-staging-test instattention-prefill-test instattention-flexgen-prefill-test instattention-flexgen-prefill-gpu-test instattention-flexgen-decode-test instattention-flexgen-decode-gpu-test instattention-flexgen-cemu-gpu-test instattention-hf-checkpoint-test instattention-tinyllama-checkpoint-test instattention-tinyllama-prefill-test instattention-tinyllama-reference-test instattention-tinyllama-decode-reference-test instattention-tinyllama-cemu-gpu-test instattention-attention-test instattention-attention-gpu-test

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

instattention-flexgen-decode-test:
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_decode_runtime.py

instattention-flexgen-decode-gpu-test:
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_decode_runtime.py --cuda

instattention-hf-checkpoint-test:
	python3 $(INSTATTENTION_DIR)/python/test_hf_checkpoint.py

instattention-tinyllama-checkpoint-test:
	python3 $(INSTATTENTION_DIR)/python/test_tinyllama_checkpoint.py \
		--model-dir "$(MODEL_DIR)" \
		$(if $(CUDA),--cuda) \
		$(if $(ALL_LAYERS),--all-layers)

instattention-tinyllama-prefill-test:
	python3 $(INSTATTENTION_DIR)/python/test_tinyllama_prefill.py \
		--model-dir "$(MODEL_DIR)" \
		--batch-size "$(if $(BATCH_SIZE),$(BATCH_SIZE),1)" \
		--prompt-length "$(if $(PROMPT_LENGTH),$(PROMPT_LENGTH),8)" \
		$(if $(CUDA),--cuda)

instattention-tinyllama-reference-test:
	python3 $(INSTATTENTION_DIR)/python/test_tinyllama_reference.py \
		--model-dir "$(MODEL_DIR)" \
		--batch-size "$(if $(BATCH_SIZE),$(BATCH_SIZE),1)" \
		--prompt-length "$(if $(PROMPT_LENGTH),$(PROMPT_LENGTH),8)" \
		--atol "$(if $(ATOL),$(ATOL),1e-4)" \
		--rtol "$(if $(RTOL),$(RTOL),1e-4)" \
		$(if $(CUDA),--cuda)

instattention-tinyllama-decode-reference-test:
	python3 $(INSTATTENTION_DIR)/python/test_tinyllama_decode_reference.py \
		--model-dir "$(MODEL_DIR)" \
		--batch-size "$(if $(BATCH_SIZE),$(BATCH_SIZE),1)" \
		--prompt-length "$(if $(PROMPT_LENGTH),$(PROMPT_LENGTH),8)" \
		--atol "$(if $(ATOL),$(ATOL),1e-4)" \
		--rtol "$(if $(RTOL),$(RTOL),1e-4)" \
		$(if $(CUDA),--cuda)

instattention-tinyllama-cemu-gpu-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_tinyllama_cemu_pipeline.py \
		--model-dir "$(MODEL_DIR)" \
		--nvm-dir "$(if $(NVM_TEST_DIR),$(NVM_TEST_DIR),/mnt/nvme0)" \
		--fdm-dir "$(if $(FDM_TEST_DIR),$(FDM_TEST_DIR),/mnt/fdm0)" \
		--csd-target "$(if $(CSD_TARGET),$(CSD_TARGET),cuda)" \
		--program "$(if $(PROGRAM),$(PROGRAM),$(if $(filter cpu,$(CSD_TARGET)),./build/dense_attention.so,./build/dense_attention_devptr.so))" \
		--batch-size "$(if $(BATCH_SIZE),$(BATCH_SIZE),1)" \
		--prompt-length "$(if $(PROMPT_LENGTH),$(PROMPT_LENGTH),8)" \
		$(if $(PROMPT_TEXT),--prompt-text "$(PROMPT_TEXT)") \
		--decode-steps "$(if $(DECODE_STEPS),$(DECODE_STEPS),1)" \
		--staging-tokens "$(if $(STAGING_TOKENS),$(STAGING_TOKENS),0)" \
		--atol "$(if $(ATOL),$(ATOL),1e-3)" \
		--rtol "$(if $(RTOL),$(RTOL),1e-3)"

instattention-flexgen-cemu-gpu-test: instattention
	python3 $(INSTATTENTION_DIR)/python/test_flexgen_cemu_pipeline.py \
		--nvm-dir "$(if $(NVM_TEST_DIR),$(NVM_TEST_DIR),/mnt/nvme0)" \
		--fdm-dir "$(if $(FDM_TEST_DIR),$(FDM_TEST_DIR),/mnt/fdm0)" \
		--decode-steps "$(if $(DECODE_STEPS),$(DECODE_STEPS),4)" \
		--num-layers "$(if $(NUM_LAYERS),$(NUM_LAYERS),2)" \
		--hidden-size "$(if $(HIDDEN_SIZE),$(HIDDEN_SIZE),16)" \
		--num-attention-heads "$(if $(QUERY_HEADS),$(QUERY_HEADS),4)" \
		--num-key-value-heads "$(if $(KV_HEADS),$(KV_HEADS),2)" \
		--staging-tokens "$(if $(STAGING_TOKENS),$(STAGING_TOKENS),0)"

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
