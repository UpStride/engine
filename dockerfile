FROM tensorflow/tensorflow:2.2.0-gpu

COPY upstride /opt/upstride/upstride
COPY setup.py /opt/upstride/setup.py
RUN cd /opt/upstride && \
    touch README.md && \
    pip install . && \
    cd / && \
    rm -r /opt/upstride
