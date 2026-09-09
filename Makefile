SHELL := /bin/bash
.DEFAULT_GOAL := help
MODEL ?= rf_detr
export MODEL

.PHONY: help doctor inspect-device prepare deploy-runtime deploy run pull decode lifecycle-dlc context-build context-run cpp-build cpp-run lifecycle-compare cpp-check
help:
	@printf '%s\n' 'make doctor                     检查 host、SDK 文件与 DLC' 'make inspect-device             检查手机属性' 'make prepare MODEL=rf_detr       准备 RF-DETR small 输入' 'make deploy-runtime             部署 QAIRT runtime 并执行 --help' 'make deploy MODEL=rf_detr        部署 RF-DETR DLC 与输入' 'make run MODEL=rf_detr           执行 RF-DETR HTP DLC online prepare' 'make pull MODEL=rf_detr          拉取并校验输出'
	@printf '%s\n' 'make decode                     解码旧单次检测输出' 'make lifecycle-dlc              DLC 建图、Finalize、重复执行与 profiling' 'make context-build              手机上预生成 context binary' 'make context-run                从 context 恢复，重复执行与 profiling' 'make cpp-build                  NDK 交叉编译最小 runtime executable' 'make cpp-run                    部署自写 runtime 并执行同一 context' 'make lifecycle-compare          校验三条路径每次输出并汇总时间' 'make cpp-check                  已部署 C++ 的错误输入与退出检查'

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

lifecycle-dlc:
	@bash scripts/lifecycle.sh dlc

context-build:
	@bash scripts/lifecycle.sh build-context

context-run:
	@bash scripts/lifecycle.sh context

cpp-build:
	@bash scripts/build-cpp.sh

cpp-run:
	@bash scripts/lifecycle.sh cpp

lifecycle-compare:
	@.venv/bin/python scripts/compare-lifecycle.py

cpp-check:
	@bash scripts/check-cpp.sh
