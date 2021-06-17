FROM tensorflow/tensorflow:2.4.1-gpu

# Upstride bash welcome screen
COPY bash.bashrc /etc/bash.bashrc
COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
COPY version /opt/upstride/version
COPY README.md /opt/upstride/README.md
RUN rm -r /opt/upstride/upstride/tests && \
    cd /opt/upstride && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
