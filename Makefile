# Makefile for GPU-CR
# Provides targets for reproducible containerized builds and local builds

GPU_VENDOR ?= NVIDIA
BUILD_DIR  ?= build
DIST_DIR   ?= dist
CUDA_IMAGE ?= nvidia/cuda:12.2.2-devel-ubuntu22.04

.PHONY: all build clean release-artifacts release-local

all: release-local

# 1. Reproducible build inside a Docker container (matches GitHub Actions environment)
release-artifacts:
	@echo "Building reproducible release artifacts in Docker ($(CUDA_IMAGE))..."
	@mkdir -p $(DIST_DIR)
	docker run --rm -u $$(id -u):$$(id -g) \
		-v $(CURDIR):/workspace -w /workspace \
		$(CUDA_IMAGE) \
		/bin/bash -c " \
			apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake build-essential && \
			cmake -B /tmp/build -S . -DCMAKE_BUILD_TYPE=Release -DGPU_VENDOR=$(GPU_VENDOR) && \
			cmake --build /tmp/build --config Release --target vGPU cr_client -j\$$(nproc) && \
			cp /tmp/build/cr_client /tmp/build/vGPU-$(GPU_VENDOR).so /workspace/$(DIST_DIR)/ && \
			cd /workspace/$(DIST_DIR) && sha256sum cr_client vGPU-$(GPU_VENDOR).so > checksums.sha256 \
		"
	@echo "Success: Release artifacts placed in $(CURDIR)/$(DIST_DIR)/:"
	@ls -lh $(DIST_DIR)/

# 2. Local CMake build (requires local CUDA + CMake installation)
release-local:
	cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=Release -DGPU_VENDOR=$(GPU_VENDOR)
	cmake --build $(BUILD_DIR) --config Release --target vGPU cr_client -j$$(nproc)

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
