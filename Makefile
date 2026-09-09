SHELL := /bin/bash
.DEFAULT_GOAL := help
MODEL ?= rf_detr
export MODEL

.PHONY: help doctor inspect-device prepare deploy-runtime deploy run pull decode
help:
	@printf '%s\n' 'make doctor                     检查 host、SDK 文件与 DLC' 'make inspect-device             检查手机属性' 'make prepare MODEL=rf_detr       准备 RF-DETR small 输入' 'make deploy-runtime             部署 QAIRT runtime 并执行 --help' 'make deploy MODEL=rf_detr        部署 RF-DETR DLC 与输入' 'make run MODEL=rf_detr           执行 RF-DETR HTP DLC online prepare' 'make pull MODEL=rf_detr          拉取并校验输出'

doctor:
	@bash scripts/doctor.sh

inspect-device:
	@bash scripts/inspect-device.sh

deploy-runtime:
	@bash scripts/deploy-runtime.sh

prepare:
	@bash scripts/prepare.sh

deploy:
	@bash scripts/deploy.sh

run:
	@bash scripts/run.sh

pull:
	@bash scripts/pull.sh

decode:
	@bash scripts/decode.sh
