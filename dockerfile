FROM tensorflow/tensorflow:2.3.0-gpu

# Upstride bash welcome screen 
COPY bash.bashrc /etc/bash.bashrc

COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
RUN cd /opt/upstride && \
    touch README.md && \
    pip install . && \
    pip install pydot graphviz && \
    cd / && \
    rm -r /opt/upstride
RUN sudo apt install graphviz -y
