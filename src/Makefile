all:
# 	/bin/bash
	python3 analyze_sweeps.py
%:
	rm -f ../results/$@/*
	python3 analyze_sweeps.py $@
