#!/bin/bash

dir_software=${HOME}/Software
dependencies="all"
num_threads=10
clean_source=false

# Parse the command line arguments
# Parse the command line arguments
for arg in "$@"; do
	case ${arg} in
	--dir-software=*)
		dir_software="${arg#--dir-software=}"
		shift
		;;
	--dependencies=*)
		dependencies="${arg#--dependencies=}"
		shift
		;;
	--num-threads=*)
		num_threads="${arg#--num-threads=}"
		shift
		;;
	--clean-source=*)
		clean_source="${arg#--clean-source=}"
		shift
		;;
	*)
		echo "Unknown argument: ${arg}"
		1
		;;
	esac
done

mkdir -p $dir_software

for str in $dependencies; do
    dependencies_lowerCase+=$(echo "$str" | tr '[:upper:]' '[:lower:]')
done
dependencies=${dependencies_lowerCase}
echo "Software directory: ${dir_software}"
echo "Dependencies: ${dependencies}"
echo "Number of threads: ${num_threads}"
echo "Clean source: ${clean_source}"

if [[ ${dependencies} == *"python3.13"* ]] || [[ ${dependencies} == "all" ]]; then
    # install python 3.13
    apt install -y build-essential zlib1g zlib1g-dev libncurses5-dev \
        libgdbm-dev pkg-config libnss3-dev libssl-dev libreadline-dev \
        libffi-dev libsqlite3-dev wget
    # install python 3.13
    cd ${dir_software} || exit
    wget -nc https://www.python.org/ftp/python/3.13.1/Python-3.13.1.tgz
    tar -xf Python-3.13.1.tgz
    cd Python-3.13.1 || exit
    ./configure --enable-optimizations
    make -j${num_threads}
    make altinstall
    echo alias python=python3.13 >>~/.bash_aliases
    echo alias pip=pip3.13 >>~/.bash_aliases
    source ${HOME}/.bashrc
fi


# install brachyutils
if [[ ${dependencies} == *"brachyutils"* ]] || [[ ${dependencies} == "all" ]]; then
    cd ${dir_software}/brachyutils || exit
    source ${HOME}/.bashrc
    python3.13 -m pip install -e .
fi
