override CONDA=$(CONDA_BASE)/bin/conda
override MAMBA=$(CONDA_BASE)/bin/mamba
override PKG=syspop
SHELL := /bin/bash

clear_env:
	rm -rf $(CONDA_BASE)/envs/$(PKG)
	$(MAMBA) index $(CONDA_BASE)/conda-bld

clear_all:
	rm -rf $(CONDA_BASE)/envs/$(PKG)
	rm -rf $(CONDA_BASE)/pkgs/$(PKG)*
	rm -rf $(CONDA_BASE)/conda-bld/linux-64/$(PKG)*
	rm -rf $(CONDA_BASE)/conda-bld/$(PKG)*
	rm -rf $(CONDA_BASE)/conda-bld/linux-64/.cache/paths/$(PKG)*
	rm -rf $(CONDA_BASE)/conda-bld/linux-64/.cache/recipe/$(PKG)*
	# $(MAMBA) index $(CONDA_BASE)/conda-bld

env: clear_all
	$(MAMBA) env create -f env.yml

upload_data:
	git lfs track "etc/data/raw_nz/**"
	git add "etc/data/raw_nz"
	git commit -m "add large data"
	git push

# -----------------------
# Publication
# -----------------------
pkg_version:
	pip index versions syspop

build_pkg:
	rm -rf $(PKG).egg*
	rm -rf dist
	python setup.py sdist bdist_wheel
	pip install .

install_twine:
	pip install twine


upload_pkg: 
	twine upload dist/*

publish: build_pkg upload_pkg

# -----------------------
# Viewer
# -----------------------
install_npm:
	source $(CONDA_BASE)/bin/activate $(PKG) & mamba install nodejs -c conda-forge 
	mkdir ~/.npm-global
	npm config set prefix '~/.npm-global'
	source $(CONDA_BASE)/bin/activate $(PKG) & export PATH=~/.npm-global/bin:$(PATH) & npm install --global http-server

run_viewer:
	~/.npm-global/bin/http-server $(DATA_DIR)

check_port:
	lsof -i -P -n


# -----------------------
# Create NZ data
# -----------------------
create_nz_data:
	etc/scripts_nz/create_nz_data.bash

# -----------------------
# Run NZ syspop
# -----------------------
run_syspop_nz:
	nohup python syspop/run_nz.py >& log.syspop &

# -----------------------
# Linting
# -----------------------
lint:
	ruff check .
