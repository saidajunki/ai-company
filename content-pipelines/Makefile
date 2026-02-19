.PHONY: all build-gion-story-gen test-gion-story-gen release-gion-story-gen

all:
	@echo 'Usage: make build-gion-story-gen | test-gion-story-gen | release-gion-story-gen'

build-gion-story-gen:
	cd ../gion-story-gen && python3 -m venv .venv && . .venv/bin/activate && pip install -q openai==0.28

test-gion-story-gen:
	cd ../gion-story-gen && . .venv/bin/activate && ./gion-story --num 1 > /dev/null

release-gion-story-gen:
	mkdir -p ../public/gion-story-gen
	cp ../gion-story-gen/gion_story_gen.py ../public/gion-story-gen/
	cp ../gion-story-gen/gion-story ../public/gion-story-gen/
	cp -r ../gion-story-gen/data ../public/gion-story-gen/
	cp ../gion-story-gen/README.md ../public/gion-story-gen/ || true
	cp ../gion-story-gen/LICENSE ../public/gion-story-gen/ || true

