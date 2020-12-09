FROM tensorflow/tensorflow:2.3.0-gpu

# Upstride bash welcome screen 
COPY bash.bashrc /etc/bash.bashrc

COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
COPY version /opt/upstride/version
RUN cd /opt/upstride && \
    touch README.md && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
