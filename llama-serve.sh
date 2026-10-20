#!/usr/bin/env bash
llama serve --temp 0.1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 -c 65000 --presence-penalty 1
