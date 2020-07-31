FROM tensorflow/tensorflow:2.3.0-gpu

# Upstride bash welcome screen 
COPY bash.bashrc /etc/bash.bashrc

COPY upstride /opt/upstride/upstride
COPY dist_binary_setup.py /opt/upstride/dist_binary_setup.py
RUN pip3 install --no-cache-dir Cython && \ 
    cd /opt/upstride && \
    python3 dist_binary_setup.py bdist_wheel && \
    pip3 install --no-cache-dir /opt/upstride/dist/*.whl && \
    cd / && \ 
    rm -r /opt/upstride