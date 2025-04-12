# Makefile for Drone Landing RL Project

# Python executable
PYTHON = python

# Script name
SCRIPT = drone_landing_rl.py

# Model architectures
MODELS = vision_transformer resnet50 simple_cnn

# Default parameters
TRAIN_EPISODES = 1000
QUICK_TRAIN_EPISODES = 300
TEST_EPISODES = 20
COMPARE_EPISODES = 500

# Default model
DEFAULT_MODEL = vision_transformer

# Output directories
CHECKPOINT_DIR = checkpoints
RESULTS_DIR = results

# Ensure directories exist
$(shell mkdir -p $(CHECKPOINT_DIR) $(RESULTS_DIR))

# Default target
.PHONY: help
help:
	@echo "Drone Landing RL Project Makefile"
	@echo "=================================="
	@echo "Available targets:"
	@echo "  help                 - Display this help message"
	@echo "  train                - Train the default model ($(DEFAULT_MODEL))"
	@echo "  train-vit            - Train the Vision Transformer model"
	@echo "  train-resnet         - Train the ResNet50 model"
	@echo "  train-cnn            - Train the Simple CNN model"
	@echo "  quick-train          - Quick training of default model (fewer episodes)"
	@echo "  quick-train-vit      - Quick training of Vision Transformer"
	@echo "  quick-train-resnet   - Quick training of ResNet50"
	@echo "  quick-train-cnn      - Quick training of Simple CNN"
	@echo "  test                 - Test the default model"
	@echo "  test-vit             - Test the Vision Transformer model"
	@echo "  test-resnet          - Test the ResNet50 model"
	@echo "  test-cnn             - Test the Simple CNN model"
	@echo "  compare              - Compare all models"
	@echo "  compare-quick        - Quick comparison of all models"
	@echo "  compare-vit-resnet   - Compare Vision Transformer and ResNet50"
	@echo "  compare-vit-cnn      - Compare Vision Transformer and Simple CNN"
	@echo "  compare-resnet-cnn   - Compare ResNet50 and Simple CNN"
	@echo "  all                  - Train and test all models"
	@echo "  clean                - Remove all checkpoint files"
	@echo ""
	@echo "Parameters:"
	@echo "  TRAIN_EPISODES=$(TRAIN_EPISODES)"
	@echo "  QUICK_TRAIN_EPISODES=$(QUICK_TRAIN_EPISODES)"
	@echo "  TEST_EPISODES=$(TEST_EPISODES)"
	@echo "  COMPARE_EPISODES=$(COMPARE_EPISODES)"

# Training targets
.PHONY: train train-vit train-resnet train-cnn
train:
	$(PYTHON) $(SCRIPT) --mode train --model $(DEFAULT_MODEL) --episodes $(TRAIN_EPISODES)

train-vit:
	$(PYTHON) $(SCRIPT) --mode train --model vision_transformer --episodes $(TRAIN_EPISODES)

train-resnet:
	$(PYTHON) $(SCRIPT) --mode train --model resnet50 --episodes $(TRAIN_EPISODES)

train-cnn:
	$(PYTHON) $(SCRIPT) --mode train --model simple_cnn --episodes $(TRAIN_EPISODES)

# Quick training targets (fewer episodes)
.PHONY: quick-train quick-train-vit quick-train-resnet quick-train-cnn
quick-train:
	$(PYTHON) $(SCRIPT) --mode train --model $(DEFAULT_MODEL) --episodes $(QUICK_TRAIN_EPISODES)

quick-train-vit:
	$(PYTHON) $(SCRIPT) --mode train --model vision_transformer --episodes $(QUICK_TRAIN_EPISODES)

quick-train-resnet:
	$(PYTHON) $(SCRIPT) --mode train --model resnet50 --episodes $(QUICK_TRAIN_EPISODES)

quick-train-cnn:
	$(PYTHON) $(SCRIPT) --mode train --model simple_cnn --episodes $(QUICK_TRAIN_EPISODES)

# Testing targets
.PHONY: test test-vit test-resnet test-cnn
test:
	$(PYTHON) $(SCRIPT) --mode test --model $(DEFAULT_MODEL) --test_episodes $(TEST_EPISODES)

test-vit:
	$(PYTHON) $(SCRIPT) --mode test --model vision_transformer --test_episodes $(TEST_EPISODES)

test-resnet:
	$(PYTHON) $(SCRIPT) --mode test --model resnet50 --test_episodes $(TEST_EPISODES)

test-cnn:
	$(PYTHON) $(SCRIPT) --mode test --model simple_cnn --test_episodes $(TEST_EPISODES)

# Model comparison targets
.PHONY: compare compare-quick compare-vit-resnet compare-vit-cnn compare-resnet-cnn
compare:
	$(PYTHON) $(SCRIPT) --mode compare --episodes $(COMPARE_EPISODES)

compare-quick:
	$(PYTHON) $(SCRIPT) --mode compare --episodes $(QUICK_TRAIN_EPISODES)

# Custom comparison scripts
compare-vit-resnet:
	@echo "Comparing Vision Transformer vs ResNet50..."
	@echo "# Vision Transformer" > $(RESULTS_DIR)/vit_vs_resnet.log
	$(PYTHON) $(SCRIPT) --mode train --model vision_transformer --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/vit_vs_resnet.log
	@echo "# ResNet50" >> $(RESULTS_DIR)/vit_vs_resnet.log
	$(PYTHON) $(SCRIPT) --mode train --model resnet50 --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/vit_vs_resnet.log
	@echo "Results saved to $(RESULTS_DIR)/vit_vs_resnet.log"

compare-vit-cnn:
	@echo "Comparing Vision Transformer vs Simple CNN..."
	@echo "# Vision Transformer" > $(RESULTS_DIR)/vit_vs_cnn.log
	$(PYTHON) $(SCRIPT) --mode train --model vision_transformer --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/vit_vs_cnn.log
	@echo "# Simple CNN" >> $(RESULTS_DIR)/vit_vs_cnn.log
	$(PYTHON) $(SCRIPT) --mode train --model simple_cnn --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/vit_vs_cnn.log
	@echo "Results saved to $(RESULTS_DIR)/vit_vs_cnn.log"

compare-resnet-cnn:
	@echo "Comparing ResNet50 vs Simple CNN..."
	@echo "# ResNet50" > $(RESULTS_DIR)/resnet_vs_cnn.log
	$(PYTHON) $(SCRIPT) --mode train --model resnet50 --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/resnet_vs_cnn.log
	@echo "# Simple CNN" >> $(RESULTS_DIR)/resnet_vs_cnn.log
	$(PYTHON) $(SCRIPT) --mode train --model simple_cnn --episodes $(COMPARE_EPISODES) >> $(RESULTS_DIR)/resnet_vs_cnn.log
	@echo "Results saved to $(RESULTS_DIR)/resnet_vs_cnn.log"

# Train and test all models
.PHONY: all
all: train-vit train-resnet train-cnn test-vit test-resnet test-cnn compare

# Custom training with specific model path testing
.PHONY: train-test-vit train-test-resnet train-test-cnn
train-test-vit:
	$(PYTHON) $(SCRIPT) --mode train --model vision_transformer --episodes $(TRAIN_EPISODES)
	$(PYTHON) $(SCRIPT) --mode test --model vision_transformer --test_episodes $(TEST_EPISODES)

train-test-resnet:
	$(PYTHON) $(SCRIPT) --mode train --model resnet50 --episodes $(TRAIN_EPISODES)
	$(PYTHON) $(SCRIPT) --mode test --model resnet50 --test_episodes $(TEST_EPISODES)

train-test-cnn:
	$(PYTHON) $(SCRIPT) --mode train --model simple_cnn --episodes $(TRAIN_EPISODES)
	$(PYTHON) $(SCRIPT) --mode test --model simple_cnn --test_episodes $(TEST_EPISODES)

# Training with specific checkpoint saving frequency
.PHONY: train-freq
train-freq:
	@echo "Set CHECKPOINT_FREQ variable (default is defined in code)"
	$(PYTHON) $(SCRIPT) --mode train --model $(DEFAULT_MODEL) --episodes $(TRAIN_EPISODES) --checkpoint_freq $(CHECKPOINT_FREQ)

# Clean up
.PHONY: clean clean-vit clean-resnet clean-cnn
clean:
	@echo "Removing all checkpoint files..."
	rm -rf $(CHECKPOINT_DIR)/*
	mkdir -p $(CHECKPOINT_DIR)

clean-vit:
	@echo "Removing Vision Transformer checkpoint files..."
	rm -rf $(CHECKPOINT_DIR)/vision_transformer/*
	mkdir -p $(CHECKPOINT_DIR)/vision_transformer

clean-resnet:
	@echo "Removing ResNet50 checkpoint files..."
	rm -rf $(CHECKPOINT_DIR)/resnet50/*
	mkdir -p $(CHECKPOINT_DIR)/resnet50

clean-cnn:
	@echo "Removing Simple CNN checkpoint files..."
	rm -rf $(CHECKPOINT_DIR)/simple_cnn/*
	mkdir -p $(CHECKPOINT_DIR)/simple_cnn

# Get system info (useful for performance tuning)
.PHONY: sysinfo
sysinfo:
	@echo "System Information:"
	@echo "---------------"
	@echo "Python Version:"
	@$(PYTHON) --version
	@echo "\nCPU Information:"
	@cat /proc/cpuinfo | grep "model name" | uniq || echo "CPU info not available"
	@echo "\nRAM Information:"
	@free -h || echo "Memory info not available"
	@echo "\nTorch Version:"
	@$(PYTHON) -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

# Show environment details
.PHONY: env
env:
	@echo "Environment Configuration:"
	@echo "CHECKPOINT_DIR = $(CHECKPOINT_DIR)"
	@echo "RESULTS_DIR = $(RESULTS_DIR)"
	@echo "TRAIN_EPISODES = $(TRAIN_EPISODES)"
	@echo "TEST_EPISODES = $(TEST_EPISODES)"
	@echo "COMPARE_EPISODES = $(COMPARE_EPISODES)"
	@echo "DEFAULT_MODEL = $(DEFAULT_MODEL)"

# Create demonstration visualization
.PHONY: visualize
visualize:
	@echo "Generating visualization of trained models..."
	$(PYTHON) $(SCRIPT) --mode test --model vision_transformer --test_episodes 3
	$(PYTHON) $(SCRIPT) --mode test --model resnet50 --test_episodes 3
	$(PYTHON) $(SCRIPT) --mode test --model simple_cnn --test_episodes 3