PROTO_DIR := ./proto
OUT_DIR := ./proto
VENV := ./venv
GEN_DIR := ./proto/gen
GRPC_PLUGIN := $(shell which grpc_python_plugin)

.PHONY: proto-gen app github lint scan

scan:
	./scripts/container-scan.sh

lint:
	./scripts/lint.sh

app:
	./scripts/run-app.sh

github: lint
	@if [ -z "$(CM)" ]; then \
		echo "Usage: make github CM=\"commit message\""; \
		exit 1; \
	fi
	@test "$$(git branch --show-current)" = main || { echo "Run make github from the main branch."; exit 1; }
	git add .
	git commit -m "$(CM)"
	git push origin main

proto-gen:
	@mkdir -p $(OUT_DIR)
	@protoc \
		-I$(PROTO_DIR) \
		--python_out=$(GEN_DIR)/ai \
		--grpc_python_out=$(GEN_DIR)/ai \
		--plugin=protoc-gen-grpc_python=$(GRPC_PLUGIN) \
		$(PROTO_DIR)/ai.proto
	@touch $(OUT_DIR)/__init__.py
	@sed -i 's/import ai_pb2 as ai__pb2/from proto.gen.ai import ai_pb2 as ai__pb2/' $(GEN_DIR)/ai/ai_pb2_grpc.py