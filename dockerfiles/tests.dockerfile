FROM eu.gcr.io/fluid-door-230710/upstride:py-1.1.0-tf2.3.0-gpu

COPY gcc-8.4.deb .

RUN \
echo Y|apt-get install dpkg \
dpkg -i gcc-8.4.deb \
update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc8.4 100  \
update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++8.4 100 \
apt-get update -y \
apt-get install cmake git -y

RUN \
git clone git@bitbucket.org:upstride/phoenix_tf.git \
cd phoenix_tf \
git submodule init \
cd core/ \
git submodule update --init --recursive  \
mkdir build && cd build \
cmake .. && make -j4 \
./tests > test.log \
# Run the following test as you want but it should be "0 failed" on that line
##tail -n 1 test.log | cut -d '|' -f 3 == '    0 failed'
cd .. && rm -rf build \
cd .. \
make \
make distclean && WITH_CUDNN=on make
