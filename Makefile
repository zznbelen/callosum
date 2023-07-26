stack=exnada/python:1.0
args=--rm -v '${PWD}'/src/:/home/exnada/src/ -v '${PWD}'/data/:/home/exnada/data/ -v '${PWD}'/results/:/home/exnada/results/
dockerrun=mkdir -p data results && docker run -it
dockerframe=${dockerrun} ${args} ${stack}
bash=/bin/bash

SHELL:=bash
OWNER:=exnada

NO_COLOR=\x1b[0m
WARN_COLOR=\x1b[0m
OK_COLOR=\x1b[32;01m
ERROR_COLOR=\x1b[31;01m
BLUE_COLOR=\x1b[34;01m
INFO_COLOR=\x1b[36;01m
bold := $(shell tput bold)
sgr0 := $(shell tput sgr0)

# 

all:
ifeq ($(shell docker images -q ${stack} 2> /dev/null),)
	@printf "$(INFO_COLOR)$(bold)Build $(stack)$(NO_COLOR)\n"
	@docker build --rm --force-rm -t $(stack) ./docker --build-arg OWNER=$(OWNER)
	@echo -n "Built image size: "
	@docker images $(stack) --format "{{.Size}}"
endif
	@${dockerframe} make -C /home/exnada/src/
	
	
container:
	@printf "$(INFO_COLOR)$(bold)Build $(stack)$(NO_COLOR)\n"
	@docker build --rm --force-rm -t $(stack) ./docker --build-arg OWNER=$(OWNER)
	@echo -n "Built image size: "
	@docker images $(stack) --format "{{.Size}}"

run:
	@${dockerframe} make -C /home/exnada/src/

%:
	@${dockerframe} make -C /home/exnada/src/ $@

bash:
	@${dockerframe} ${bash}

bashroot:
	@${dockerrun} --user root ${args} ${stack} ${bash}

clean:
	@docker image rm $(stack)