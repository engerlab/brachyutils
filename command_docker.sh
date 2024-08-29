command_list="build,save,run"
# sudo usermod -aG docker $USER
if [[ "$#" -eq 1 ]]; then
    command_list="${1#--command-list=}"
fi

if [[ $command_list == *"build"* ]]; then
    # build the docker image
    docker image build\
        -t brachyutils \
        -f Dockerfile \
        .
    # one liner
    # docker image build -t rapidbrachy -f docker_src/Dockerfile .
fi
if [[ $command_list == *"save"* ]]; then
    # save the docker image to a tar file
    docker save\
        -o brachyutils.tar\
        brachyutils:latest
    tar -cvfz brachyutils.tar.gz brachyutils.tar
fi

if [[ $command_list == *"run"* ]]; then
    # run the docker image
    docker run --rm \
        -it -v $(pwd):/root/brachyutils \
        brachyutils:latest /bin/bash
fi